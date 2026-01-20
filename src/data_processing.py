"""
Data processing utilities for FIM and comment-to-code training.
Shared across all notebooks for consistency.
"""
from __future__ import annotations

import json
import os
import random
import re
import ast
from typing import List, Dict, Optional, Tuple, Iterable, Any
from dataclasses import dataclass
from pathlib import Path
import hashlib

import torch


@dataclass(frozen=True)
class FimTokens:
    """FIM special tokens for Qwen models."""
    prefix: str = "<|fim_prefix|>"
    middle: str = "<|fim_middle|>"
    suffix: str = "<|fim_suffix|>"
    pad: str = "<|fim_pad|>"


FIM = FimTokens()


def tokenizer_supports_fim(tokenizer) -> bool:
    """Check if tokenizer has FIM tokens in vocabulary."""
    if tokenizer is None:
        return False

    try:
        vocab = tokenizer.get_vocab()
        return (FIM.prefix in vocab) and (FIM.middle in vocab) and (FIM.suffix in vocab)
    except Exception:
        pass

    try:
        ids = tokenizer.convert_tokens_to_ids([FIM.prefix, FIM.middle, FIM.suffix])
        return all(isinstance(i, int) and i >= 0 for i in ids)
    except Exception:
        return False

def format_fim_text(prefix: str, suffix: str, middle: str) -> str:
    """
    Canonical FIM text format used everywhere (training + inference):
      <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
    """
    return f"{FIM.prefix}{prefix}{FIM.suffix}{suffix}{FIM.middle}{middle}"

def build_fim_text(ex: Dict[str, Any]) -> str:
    if ex.get('text'):
        return ex['text']
    return format_fim_text(ex.get('prefix',''), ex.get('suffix',''), ex.get('middle',''))

def build_autocomplete_prompt(prefix: str, suffix: Optional[str], tokenizer) -> str:
    """
    Build FIM prompt for autocomplete (runtime/inference).
    """
    if suffix and tokenizer_supports_fim(tokenizer):
        return f"{FIM.prefix}{prefix}{FIM.suffix}{suffix}{FIM.middle}"
    return prefix

def build_c2c_prompt_completion(ex: Dict[str, Any]) -> Tuple[str, str]:
    # Prefer structured fields
    prompt = (ex.get('prompt') or '')
    completion = (ex.get('completion') or '')
    if prompt or completion:
        return prompt, completion
    # Fallback: try to split from text if needed
    text = ex.get('text') or ''
    return '', text


@dataclass
class Comment2CodeCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for ex in batch:
            prompt, completion = build_c2c_prompt_completion(ex)
            prompt = prompt or ''
            completion = completion or ''

            # Tokenize prompt & completion separately for correct masking
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).get('input_ids', [])

            # Ensure we leave room for completion
            remaining = max(self.max_length - len(prompt_ids), 0)
            comp_ids = self.tokenizer(completion, add_special_tokens=False, truncation=True, max_length=remaining).get('input_ids', [])

            ids = (prompt_ids + comp_ids)[: self.max_length]
            labels = ([-100] * min(len(prompt_ids), self.max_length)) + comp_ids
            labels = labels[: self.max_length]

            # If nothing to learn, create a 1-token no-loss sample
            if len(ids) == 0 or all(l == -100 for l in labels):
                ids = [self.tokenizer.eos_token_id]
                labels = [-100]

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class FimCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for ex in batch:
            text = build_fim_text(ex)
            ids = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.max_length).get('input_ids', [])
            if len(ids) == 0:
                ids = [self.tokenizer.eos_token_id]
                labels = [-100]
            else:
                labels = ids[:]  # standard LM
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class MixedDispatchCollator:
    c2c: Comment2CodeCollator
    fim: FimCollator

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # We collate sample-by-sample to keep logic simple and correct.
        # (If you want speed, you can split batch by task and merge, but this is robust.)
        out_list = []
        for ex in batch:
            if ex.get('task') == 'comment2code':
                out_list.append(self.c2c([ex]))
            else:
                out_list.append(self.fim([ex]))

        # Merge singleton batches into one batch
        input_ids = torch.cat([o['input_ids'] for o in out_list], dim=0)
        attention_mask = torch.cat([o['attention_mask'] for o in out_list], dim=0)
        labels = torch.cat([o['labels'] for o in out_list], dim=0)

        # Now pad to max length among these singletons
        pad_id = self.c2c.tokenizer.pad_token_id
        # pad_sequence expects list of 1D tensors
        input_ids_list = [t.squeeze(0) for t in input_ids.split(1, dim=0)]
        labels_list = [t.squeeze(0) for t in labels.split(1, dim=0)]
        attn_list = [t.squeeze(0) for t in attention_mask.split(1, dim=0)]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ============================================================================
# Deterministic Hashing / IDs
# ============================================================================

def stable_int_hash(text: str, *, mod: int = 2**31 - 1) -> int:
    """
    Stable integer hash for deterministic seeding across runs/machines.  Uses SHA256 -> int -> mod.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % mod  # 64-bit slice then mod into int range


def stable_example_id(language: str, code: str, hint: Optional[str] = None) -> str:
    """
    Stable id used for per-example seed derivation and provenance.
    """
    base = f"{language}\n{hint or ''}\n{code}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


# ============================================================================
# FIM Training Data Generation
# ============================================================================

def _newline_positions(code: str) -> List[int]:
    """Get positions of all newlines in code."""
    return [i for i, ch in enumerate(code) if ch == "\n"]


def _pick_two_cuts(
    code: str, 
    rng: random.Random,
    min_length: int = 200,
    min_gap: int = 40,
    min_lines: int = 10
) -> Optional[Tuple[int, int]]:
    """
    Choose two cut points (i < j) biased toward newline boundaries.
    Return a Tuple of (i, j) positions or None if unsuitable
    """
    if len(code) < min_length:
        return None
        
    nl = _newline_positions(code)
    if len(nl) < min_lines:
        return None

    # Avoid extreme ends
    safe_start = max(2, len(nl) // 10)
    safe_end = min(len(nl) - 2, len(nl) - len(nl) // 10)
    
    if safe_end - safe_start < 6:
        return None
    
    i = rng.choice(nl[safe_start:safe_end - 3])
    j = rng.choice(nl[safe_start + 3:safe_end])
    
    if not (0 < i < j < len(code)):
        return None
    if (j - i) < min_gap:
        return None
        
    return i, j


def make_fim_sample(prefix: str, suffix: str, middle: str) -> str:
    """Create FIM training sample in correct format."""
    return f"{FIM.prefix}{prefix}{FIM.suffix}{suffix}{FIM.middle}{middle}"


def code_to_fim_samples(
    code: str,
    language: str,
    n_samples: int = 4,
    seed: int = 42,
    example_id: Optional[str] = None,
    include_text: bool = True,
    min_length: int = 200,
    min_gap: int = 40,
    min_lines: int = 10,
) -> List[Dict]:
    """
    Generate FIM training samples from raw code.

    IMPORTANT FIX:
    - We derive a per-example seed from global seed + stable example_id to avoid
      generating the same cut patterns for every example.
    """
    lang = normalize_language(language)
    code = (code or "").strip("\n")
    if not code.strip():
        return []

    ex_id = example_id or stable_example_id(lang, code)
    ex_seed = stable_int_hash(f"{seed}:{ex_id}")
    rng = random.Random(ex_seed)

    samples: List[Dict] = []

    for k in range(n_samples):
        cuts = _pick_two_cuts(
            code,
            rng,
            min_length=min_length,
            min_gap=min_gap,
            min_lines=min_lines,
        )
        if not cuts:
            continue

        i, j = cuts
        prefix = code[:i]
        middle = code[i:j]
        suffix = code[j:]

        sample = {
            "task": "autocomplete_fim",
            "language": lang,
            "id": f"{ex_id}:fim:{k}",
            "prefix": prefix,
            "suffix": suffix,
            "middle": middle,
        }
        if include_text:
            sample["text"] = format_fim_text(prefix, suffix, middle)

        samples.append(sample)

    return samples


# ============================================================================
# Comment-to-Code Data Generation
# ============================================================================

_LANG_MAP = {
    "py": "python",
    "python": "python",
    "ts": "typescript",
    "typescript": "typescript",
    "js": "javascript",
    "javascript": "javascript",
}


def normalize_language(lang: str) -> str:
    """Normalize language name."""
    if not lang:
        return "python"
    return _LANG_MAP.get(lang.strip().lower(), lang.strip().lower())


def strip_wrapping_doc_delims(text: str) -> str:
    """Remove ONLY wrapping doc delimiters, not interior content."""
    if not text:
        return ""

    t = text.strip()

    # Python triple-quote wrappers
    if (t.startswith('"""') and t.endswith('"""')) or (t.startswith("'''") and t.endswith("'''")):
        t = t[3:-3].strip()

    # JSDoc / block comment wrappers
    if t.startswith("/**") and t.endswith("*/"):
        t = t[3:-2].strip()
    elif t.startswith("/*") and t.endswith("*/"):
        t = t[2:-2].strip()

    # Remove leading '*' common in JSDoc blocks
    t = re.sub(r"^\s*\*\s?", "", t, flags=re.MULTILINE).strip()
    return t


def squash_blank_lines(text: str, max_consecutive: int = 1) -> str:
    """Reduce consecutive blank lines."""
    if not text:
        return ""
    out = []
    blanks = 0
    for line in text.splitlines():
        if line.strip() == "":
            blanks += 1
            if blanks <= max_consecutive:
                out.append("")
        else:
            blanks = 0
            out.append(line.rstrip())
    return "\n".join(out).strip()


def remove_leading_doc_block_from_code(code: str, doc: str, language: str) -> str:
    """
    Remove leading doc block from code if it duplicates the docstring.
    """
    if not code or not doc:
        return code

    c = code.lstrip()
    d = doc.strip()

    if language == "python":
        # Leading triple-quoted docstring
        m = re.match(r'^(?:[rRuU]?)(?:"""|\'\'\')([\s\S]*?)(?:"""|\'\'\')\s*', c)
        if m:
            inner = strip_wrapping_doc_delims(m.group(0))
            if d in inner or inner in d:
                return c[m.end():].lstrip()

    else:  # js/ts
        # Leading /* ... */ or /** ... */
        m = re.match(r"^/\*\*?([\s\S]*?)\*/\s*", c)
        if m:
            inner = strip_wrapping_doc_delims(m.group(0))
            if d in inner or inner in d:
                return c[m.end():].lstrip()

    return code


def build_comment2code_prefix(language: str, doc: str) -> str:
    """Build instruction prefix for comment-to-code training."""
    lang = normalize_language(language)
    doc = (doc or "").strip()
    
    if lang == "python":
        return "\n".join([
            "# Write Python code for the following docstring",
            '"""',
            doc,
            '"""',
            "",
        ])
    else:
        return "\n".join([
            "// Write TypeScript/JavaScript code for the following comment",
            "/**",
            doc,
            "*/",
            "",
        ])


def clean_code_doc_pair(example: Dict) -> Tuple[str, str, str]:
    """
    Clean and prepare code/doc pair for training.
    
    Args:
        example: Dict with 'language', 'input' (doc), 'output' (code)
        
    Returns:
        Tuple of (language, clean_doc, clean_code)
    """
    language = normalize_language(example.get("language", "python"))
    doc_raw = (example.get("input") or "").strip()
    code_raw = (example.get("output") or "").strip()

    # Clean doc/comment text
    doc_clean = strip_wrapping_doc_delims(doc_raw)
    doc_clean = squash_blank_lines(doc_clean, max_consecutive=1)

    # Clean code
    code_clean = remove_leading_doc_block_from_code(code_raw, doc_clean, language)
    code_clean = squash_blank_lines(code_clean, max_consecutive=1)

    return (language, doc_clean, code_clean)


def create_comment2code_sample(
    example: Dict,
    *,
    include_text: bool = True,
    example_id: Optional[str] = None,
) -> Dict:
    """
    Create comment-to-code training sample.
    
    Args:
        example: Dict with 'language', 'input', 'output'
        
    Returns:
        Dict with 'text', 'task', 'language'
    """
    lang, doc, code = clean_code_doc_pair(example)
    prompt = build_comment2code_prefix(lang, doc)
    completion = code

    ex_id = example_id or stable_example_id(lang, completion, hint=doc)

    sample = {
        "task": "comment2code",
        "language": lang,
        "id": f"{ex_id}:c2c",
        "prompt": prompt,
        "completion": completion,
    }
    if include_text:
        sample["text"] = prompt + completion  # some trainers still want a single field

    return sample


# ============================================================================
# Dataset Mixing
# ============================================================================

def create_mixed_dataset(
    code_examples: List[Dict],
    fim_ratio: float = 0.8,
    fim_samples_per_code: int = 4,
    seed: int = 42,
    include_text: bool = True,
) -> List[Dict]:
    """
    Create mixed FIM + comment-to-code dataset.

    Args:
        code_examples: List of dicts with 'language', 'input', 'output'
        fim_ratio: Ratio of FIM samples (0.8 = 80% FIM, 20% comment-to-code)
        fim_samples_per_code: Number of FIM samples per code example
        seed: Random seed

    Returns:
        List of mixed training samples
    """
    if not (0.0 <= fim_ratio <= 1.0):
        raise ValueError(f"fim_ratio must be in [0,1], got {fim_ratio}")

    rng = random.Random(seed)

    fim_pool: List[Dict] = []
    c2c_pool: List[Dict] = []

    for idx, ex in enumerate(code_examples):
        lang = normalize_language(ex.get("language", "python"))
        code = (ex.get("output") or "").strip()
        doc = (ex.get("input") or "").strip()

        if not code:
            continue

        # stable id for this raw example; prefer path/id if provided
        hint = ex.get("id") or ex.get("path") or f"idx={idx}"
        ex_id = stable_example_id(lang, code, hint=hint)

        fim_pool.extend(
            code_to_fim_samples(
                code=code,
                language=lang,
                n_samples=fim_samples_per_code,
                seed=seed,
                example_id=ex_id,
                include_text=include_text,
            )
        )
        c2c_pool.append(
            create_comment2code_sample(
                {"language": lang, "input": doc, "output": code},
                include_text=include_text,
                example_id=ex_id,
            )
        )

    if not fim_pool and not c2c_pool:
        return []

    # Decide targets based on pool sizes (not on already-mixed list)
    max_total = len(fim_pool) + len(c2c_pool)
    fim_target = int(max_total * fim_ratio)
    c2c_target = max_total - fim_target

    # Clamp to pool sizes
    fim_selected = rng.sample(fim_pool, k=min(fim_target, len(fim_pool))) if fim_pool else []
    c2c_selected = rng.sample(c2c_pool, k=min(c2c_target, len(c2c_pool))) if c2c_pool else []

    mixed = fim_selected + c2c_selected
    rng.shuffle(mixed)

    return mixed

# ===========================================================================
# Helpers for code extraction
# ==========================================================================

def extract_cdk_services(code: str) -> List[str]:
    """Extract AWS services mentioned in the code."""
    services = []

    # Common CDK service patterns
    service_patterns = {
        # Core compute / storage / database
        'S3': [
            's3.Bucket', 'aws_s3', 's3.', 'Bucket(', 'aws-s3',
            'aws-cdk-lib/aws-s3', 'aws_s3_assets', 's3_assets.', 'Asset(',
            'BucketPolicy(', 'CfnBucket', 'CfnBucketPolicy'
        ],
        'Lambda': [
            'lambda_.Function', 'aws_lambda', 'lambda.', 'Function(', 'aws-lambda',
            'aws-cdk-lib/aws-lambda', 'lambda.Function', 'lambda_.LayerVersion',
            'LayerVersion(', 'Code.from', 'InlineCode(', 'DockerImageFunction(',
            'CfnFunction', 'CfnPermission', 'Alias(', 'Version('
        ],
        'DynamoDB': [
            'dynamodb.Table', 'aws_dynamodb', 'dynamodb.', 'Table(', 'aws-dynamodb',
            'aws-cdk-lib/aws-dynamodb', 'dynamodb.Table', 'CfnTable',
            'AttributeType.', 'BillingMode.', 'StreamViewType.', 'GlobalSecondaryIndex'
        ],
        'RDS': [
            'rds.Database', 'aws_rds', 'rds.', 'aws-rds',
            'aws-cdk-lib/aws-rds', 'DatabaseInstance(', 'DatabaseCluster(',
            'ServerlessCluster(', 'Aurora', 'CfnDBInstance', 'CfnDBCluster',
            'ParameterGroup(', 'OptionGroup('
        ],
        'EC2': [
            'ec2.', 'aws_ec2', 'aws-ec2', 'Instance(', 'SecurityGroup(',
            'Subnet(', 'SubnetType.', 'Port.', 'Peer.', 'CfnInstance', 'CfnSecurityGroup'
        ],
        'VPC': [
            'ec2.Vpc', 'aws_ec2', 'Vpc(', 'vpc.', 'aws-ec2',
            'aws-cdk-lib/aws-ec2', 'Vpc.from', 'SubnetConfiguration', 'NatProvider',
            'CfnVPC', 'CfnSubnet', 'CfnRouteTable', 'CfnRoute', 'CfnInternetGateway',
            'CfnNatGateway', 'CfnVPCEndpoint'
        ],

        # Identity / security
        'IAM': [
            'iam.Role', 'aws_iam', 'iam.', 'Role(', 'Policy(', 'aws-iam',
            'aws-cdk-lib/aws-iam', 'ManagedPolicy', 'PolicyStatement', 'Effect.',
            'ServicePrincipal(', 'AccountPrincipal(', 'ArnPrincipal(',
            'CfnRole', 'CfnPolicy', 'CfnManagedPolicy'
        ],
        'KMS': [
            'kms.Key', 'aws_kms', 'Key(', 'aws-kms',
            'aws-cdk-lib/aws-kms', 'Key.from', 'Alias(', 'CfnKey', 'CfnAlias'
        ],
        'Secrets Manager': [
            'secretsmanager.', 'aws_secretsmanager', 'aws-secretsmanager',
            'aws-cdk-lib/aws-secretsmanager', 'Secret(', 'Secret.from',
            'CfnSecret', 'SecretStringGenerator'
        ],
        'SSM Parameter Store': [
            'ssm.', 'aws_ssm', 'aws-ssm',
            'aws-cdk-lib/aws-ssm', 'StringParameter(', 'StringParameter.from',
            'CfnParameter'
        ],
        'Cognito': [
            'cognito.', 'aws_cognito', 'aws-cognito',
            'aws-cdk-lib/aws-cognito', 'UserPool(', 'UserPoolClient(',
            'IdentityPool(', 'CfnUserPool', 'CfnUserPoolClient'
        ],
        'WAF': [
            'wafv2.', 'aws_wafv2', 'aws-wafv2', 'aws-waf',
            'aws-cdk-lib/aws-wafv2', 'CfnWebACL', 'WebAcl', 'Rule'
        ],

        # API / integration / messaging
        'API Gateway': [
            'apigateway.', 'aws_apigateway', 'RestApi(', 'aws-apigateway',
            'aws-cdk-lib/aws-apigateway', 'HttpApi(', 'WebSocketApi(',
            'LambdaIntegration', 'MockIntegration', 'CfnRestApi', 'CfnStage', 'CfnDeployment'
        ],
        'AppSync': [
            'appsync.', 'aws_appsync', 'aws-appsync',
            'aws-cdk-lib/aws-appsync', 'GraphqlApi(', 'SchemaFile(',
            'CfnGraphqlApi', 'CfnApiKey', 'CfnDataSource'
        ],
        'EventBridge': [
            'events.', 'aws_events', 'aws-eventbridge', 'aws-events',
            'aws-cdk-lib/aws-events', 'Rule(', 'EventBus(', 'CfnRule', 'CfnEventBus'
        ],
        'SNS': [
            'sns.Topic', 'aws_sns', 'Topic(', 'aws-sns',
            'aws-cdk-lib/aws-sns', 'Subscription(', 'CfnTopic', 'CfnSubscription'
        ],
        'SQS': [
            'sqs.Queue', 'aws_sqs', 'Queue(', 'aws-sqs',
            'aws-cdk-lib/aws-sqs', 'DeadLetterQueue', 'CfnQueue'
        ],
        'Kinesis': [
            'kinesis.', 'aws_kinesis', 'aws-kinesis',
            'aws-cdk-lib/aws-kinesis', 'Stream(', 'CfnStream', 'StreamMode.'
        ],
        'MSK (Kafka)': [
            'msk.', 'aws_msk', 'aws-msk',
            'aws-cdk-lib/aws-msk', 'Cluster(', 'CfnCluster'
        ],
        'Step Functions': [
            'stepfunctions.', 'aws_stepfunctions', 'aws-stepfunctions',
            'aws-cdk-lib/aws-stepfunctions', 'StateMachine(', 'Chain', 'Pass(', 'TaskInput',
            'CfnStateMachine'
        ],
        'SFn Tasks': [
            'stepfunctions_tasks.', 'aws_stepfunctions_tasks',
            'aws-cdk-lib/aws-stepfunctions-tasks', 'LambdaInvoke(', 'EcsRunTask(',
            'GlueStartJobRun(', 'SqsSendMessage(', 'SnsPublish('
        ],

        # Observability
        'CloudWatch': [
            'cloudwatch.', 'aws_cloudwatch', 'aws-cloudwatch',
            'aws-cdk-lib/aws-cloudwatch', 'Metric(', 'Alarm(', 'Dashboard(',
            'CfnAlarm', 'CfnDashboard'
        ],
        'CloudWatch Logs': [
            'logs.', 'aws_logs', 'aws-logs',
            'aws-cdk-lib/aws-logs', 'LogGroup(', 'RetentionDays.',
            'CfnLogGroup', 'MetricFilter('
        ],
        'X-Ray': [
            'xray.', 'aws_xray', 'aws-xray',
            'aws-cdk-lib/aws-xray', 'CfnGroup', 'CfnSamplingRule'
        ],

        # IaC / deployment tooling
        'CloudFormation': [
            'cloudformation.', 'aws_cloudformation', 'aws-cloudformation',
            'aws-cdk-lib/aws-cloudformation', 'Stack', 'CfnStack', 'CfnWaitCondition'
        ],
        'CDK Pipelines': [
            'pipelines.', 'aws_pipelines', 'aws-cdk-lib/pipelines',
            'CodePipeline(', 'ShellStep(', 'CodeBuildStep('
        ],
        'CodePipeline': [
            'codepipeline.', 'aws_codepipeline', 'aws-codepipeline',
            'aws-cdk-lib/aws-codepipeline', 'Pipeline(', 'CfnPipeline'
        ],
        'CodeBuild': [
            'codebuild.', 'aws_codebuild', 'aws-codebuild',
            'aws-cdk-lib/aws-codebuild', 'Project(', 'BuildSpec', 'CfnProject'
        ],
        'CodeDeploy': [
            'codedeploy.', 'aws_codedeploy', 'aws-codedeploy',
            'aws-cdk-lib/aws-codedeploy', 'ServerApplication(', 'LambdaApplication(',
            'CfnApplication', 'CfnDeploymentGroup'
        ],

        # Containers
        'ECS': [
            'ecs.', 'aws_ecs', 'aws-ecs',
            'aws-cdk-lib/aws-ecs', 'Cluster(', 'FargateService(', 'Ec2Service(',
            'TaskDefinition(', 'ContainerImage', 'CfnService', 'CfnCluster', 'CfnTaskDefinition'
        ],
        'ECR': [
            'ecr.', 'aws_ecr', 'aws-ecr',
            'aws-cdk-lib/aws-ecr', 'Repository(', 'CfnRepository'
        ],
        'EKS': [
            'eks.', 'aws_eks', 'aws-eks',
            'aws-cdk-lib/aws-eks', 'Cluster(', 'KubernetesManifest(', 'HelmChart(',
            'CfnCluster', 'CfnAddon'
        ],

        # Networking / edge
        'ELB / ALB / NLB': [
            'elasticloadbalancingv2.', 'aws_elasticloadbalancingv2', 'aws-elasticloadbalancingv2',
            'aws-cdk-lib/aws-elasticloadbalancingv2', 'ApplicationLoadBalancer(',
            'NetworkLoadBalancer(', 'Listener(', 'TargetGroup(', 'CfnLoadBalancer'
        ],
        'Route 53': [
            'route53.', 'aws_route53', 'aws-route53',
            'aws-cdk-lib/aws-route53', 'HostedZone(', 'ARecord(', 'CnameRecord(',
            'CfnHostedZone', 'CfnRecordSet'
        ],
        'ACM': [
            'certificatemanager.', 'aws_certificatemanager', 'aws-acm',
            'aws-cdk-lib/aws-certificatemanager', 'Certificate(', 'DnsValidatedCertificate(',
            'CfnCertificate'
        ],
        'CloudFront': [
            'cloudfront.', 'aws_cloudfront', 'aws-cloudfront',
            'aws-cdk-lib/aws-cloudfront', 'Distribution(', 'CloudFrontWebDistribution(',
            'OriginAccessIdentity', 'CfnDistribution'
        ],
        'S3 + CloudFront Origins': [
            'origins.', 'aws_cloudfront_origins',
            'aws-cdk-lib/aws-cloudfront-origins', 'S3Origin(', 'HttpOrigin(', 'RestApiOrigin('
        ],
        'Global Accelerator': [
            'globalaccelerator.', 'aws_globalaccelerator', 'aws-globalaccelerator',
            'aws-cdk-lib/aws-globalaccelerator', 'Accelerator(', 'CfnAccelerator'
        ],

        # Data / analytics
        'Redshift': [
            'redshift.Cluster', 'aws_redshift', 'Cluster(', 'aws-redshift',
            'aws-cdk-lib/aws-redshift', 'CfnCluster', 'CfnClusterSubnetGroup'
        ],
        'Athena': [
            'athena.', 'aws_athena', 'aws-athena',
            'aws-cdk-lib/aws-athena', 'CfnWorkGroup', 'CfnNamedQuery'
        ],
        'Glue': [
            'glue.', 'aws_glue', 'aws-glue',
            'aws-cdk-lib/aws-glue', 'Job(', 'CfnJob', 'CfnCrawler', 'Database(', 'CfnDatabase'
        ],
        'EMR': [
            'emr.', 'aws_emr', 'aws-emr',
            'aws-cdk-lib/aws-emr', 'CfnCluster', 'CfnStep', 'CfnInstanceGroupConfig'
        ],
        'Lake Formation': [
            'lakeformation.', 'aws_lakeformation', 'aws-lakeformation',
            'aws-cdk-lib/aws-lakeformation', 'CfnPermissions', 'CfnResource'
        ],
        'OpenSearch (Elasticsearch)': [
            'opensearchservice.', 'aws_opensearchservice', 'aws-opensearchservice',
            'elasticsearch.', 'aws_elasticsearch', 'aws-elasticsearch',
            'aws-cdk-lib/aws-opensearchservice', 'Domain(', 'CfnDomain'
        ],

        # Caching / files
        'ElastiCache': [
            'elasticache.', 'aws_elasticache', 'aws-elasticache',
            'aws-cdk-lib/aws-elasticache', 'CfnCacheCluster', 'CfnReplicationGroup', 'CfnSubnetGroup'
        ],
        'EFS': [
            'efs.FileSystem', 'aws_efs', 'FileSystem(', 'aws-efs',
            'aws-cdk-lib/aws-efs', 'AccessPoint(', 'CfnFileSystem', 'CfnAccessPoint'
        ],
        'FSx': [
            'fsx.', 'aws_fsx', 'aws-fsx',
            'aws-cdk-lib/aws-fsx', 'LustreFileSystem(', 'WindowsFileSystem(',
            'CfnFileSystem'
        ],

        # Dev / ops
        'CloudTrail': [
            'cloudtrail.', 'aws_cloudtrail', 'aws-cloudtrail',
            'aws-cdk-lib/aws-cloudtrail', 'Trail(', 'CfnTrail'
        ],
        'Config': [
            'config.', 'aws_config', 'aws-config',
            'aws-cdk-lib/aws-config', 'CfnConfigurationRecorder', 'CfnDeliveryChannel', 'ManagedRule'
        ],
        'Backup': [
            'backup.', 'aws_backup', 'aws-backup',
            'aws-cdk-lib/aws-backup', 'BackupVault(', 'BackupPlan(', 'CfnBackupVault'
        ],

        # ML
        'SageMaker': [
            'sagemaker.', 'aws_sagemaker', 'aws-sagemaker',
            'aws-cdk-lib/aws-sagemaker', 'CfnEndpoint', 'CfnEndpointConfig', 'CfnModel',
            'CfnTrainingJob', 'CfnNotebookInstance'
        ],

        # Misc common
        'CloudMap (Service Discovery)': [
            'servicediscovery.', 'aws_servicediscovery', 'aws-servicediscovery',
            'aws-cdk-lib/aws-servicediscovery', 'PrivateDnsNamespace(', 'Service(',
            'CfnService'
        ],
    }

    for service, patterns in service_patterns.items():
        if any(pattern in code for pattern in patterns):
            services.append(service)

    return services

def get_preceding_comment_block(
    lines: List[str],
    start_line_idx: int,
    *,
    lang: str,
    max_lines: int = 20,
    min_chars: int = 20,
    allow_blank_within: int = 2,
) -> Optional[str]:
    """
    Extract a human-readable comment block immediately preceding a node starting at `start_line_idx`.
    """

    _PY_COMMENT_RE = re.compile(r"^\s*#(.*)$")
    _TS_LINE_COMMENT_RE = re.compile(r"^\s*//(.*)$")
    _TS_BLOCK_START_RE = re.compile(r"^\s*/\*\*?")  # /* or /**
    _TS_BLOCK_END_RE = re.compile(r".*\*/\s*$")

    lang_norm = lang.strip().lower()
    if lang_norm in {"python"}:
        lang_norm = "py"
    if lang_norm in {"typescript", "javascript", "js"}:
        lang_norm = "ts"

    def _clean_common(text: str) -> str:
        # Collapse whitespace, strip bullets, etc.
        text = text.strip()
        # Remove leading common bullet markers
        text = re.sub(r"^\s*[-*]+\s*", "", text)
        # Collapse internal whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_noise(line: str) -> bool:
        s = line.strip().lower()
        if not s:
            return True
        # common low-signal markers
        return s in {"todo", "fixme"} or s.startswith(("todo:", "fixme:"))

    # ---------- Python: # comments ----------
    if lang_norm == "py":
        i = start_line_idx - 1
        buf: List[str] = []
        blanks = 0
        while i >= 0 and (start_line_idx - 1 - i) < max_lines:
            raw = lines[i].rstrip("\n")
            if raw.strip() == "":
                blanks += 1
                if blanks > allow_blank_within:
                    break
                i -= 1
                continue

            m = _PY_COMMENT_RE.match(raw)
            if m:
                blanks = 0
                txt = m.group(1).strip()
                if txt and not _is_noise(txt):
                    buf.append(txt)
                i -= 1
                continue

            # stop at first non-comment non-blank
            break

        buf = list(reversed(buf))
        text = _clean_common(" ".join(buf))
        return text if len(text) >= min_chars else None

    # ---------- TS/JS: // comments or /* ... */ ----------
    if lang_norm == "ts":
        i = start_line_idx - 1

        # First: try to capture a /* ... */ block immediately above
        # We scan upwards to find the end '*/' first; if present, collect until '/*' or '/**'.
        scan_limit = max(0, start_line_idx - max_lines)
        blanks = 0

        # Skip a couple blanks above the node
        while i >= scan_limit and lines[i].strip() == "":
            blanks += 1
            if blanks > allow_blank_within:
                break
            i -= 1

        # If we land on a line that ends a block comment, collect it.
        if i >= scan_limit and _TS_BLOCK_END_RE.match(lines[i]):
            block_lines: List[str] = []
            j = i
            while j >= scan_limit:
                block_lines.append(lines[j].rstrip("\n"))
                if _TS_BLOCK_START_RE.match(lines[j]):
                    break
                j -= 1

            if j >= scan_limit and _TS_BLOCK_START_RE.match(lines[j]):
                block_lines = list(reversed(block_lines))

                # Strip /*, */, and leading * in JSDoc blocks
                cleaned: List[str] = []
                for ln in block_lines:
                    ln = re.sub(r"^\s*/\*\*?\s*", "", ln)
                    ln = re.sub(r"\*/\s*$", "", ln)
                    ln = re.sub(r"^\s*\*\s?", "", ln)
                    ln = ln.strip()
                    if ln and not _is_noise(ln):
                        cleaned.append(ln)

                text = _clean_common(" ".join(cleaned))
                if len(text) >= min_chars:
                    return text

            # If we saw '*/' but didn't find a proper start in window, fall through to // mode.

        # Second: collect consecutive // lines
        i = start_line_idx - 1
        buf2: List[str] = []
        blanks = 0
        while i >= scan_limit:
            raw = lines[i].rstrip("\n")
            if raw.strip() == "":
                blanks += 1
                if blanks > allow_blank_within:
                    break
                i -= 1
                continue

            m = _TS_LINE_COMMENT_RE.match(raw)
            if m:
                blanks = 0
                txt = m.group(1).strip()
                if txt and not _is_noise(txt):
                    buf2.append(txt)
                i -= 1
                continue

            break

        buf2 = list(reversed(buf2))
        text = _clean_common(" ".join(buf2))
        return text if len(text) >= min_chars else None

    raise ValueError(f"Unsupported lang: {lang!r} (use 'py' or 'ts')")

def generate_python_docstring(node, code: str) -> str:
    """Generate meaningful docstrings for Python CDK constructs."""
    name = node.name
    services = extract_cdk_services(code)

    if isinstance(node, ast.ClassDef):
        if 'Stack' in name:
            if services:
                return f"CDK Stack that creates {', '.join(services[:3])} resources"
            else:
                return f"CDK Stack for AWS infrastructure deployment"
        elif 'Construct' in name:
            if services:
                return f"CDK Construct for {', '.join(services[:2])} infrastructure components"
            else:
                return f"CDK Construct {name} for reusable infrastructure components"
        else:
            return f"CDK class {name} for AWS resource management"
    else:
        # Function
        if 'lambda' in name.lower() or 'handler' in name.lower():
            return f"Lambda function handler {name}"
        elif services:
            return f"CDK helper function for {', '.join(services[:2])} operations"
        else:
            return f"CDK helper function {name}"

def generate_typescript_docstring(name: str, code: str, construct_type: str) -> str:
    """Generate meaningful docstrings for TypeScript CDK constructs."""
    services = extract_cdk_services(code)

    if construct_type == 'class':
        if 'Stack' in name:
            if services:
                return f"CDK Stack that creates {', '.join(services[:4])} resources"
            else:
                return f"CDK Stack for AWS infrastructure deployment"
        elif 'Construct' in name:
            if services:
                return f"CDK Construct for {', '.join(services[:3])} infrastructure components"
            else:
                return f"CDK Construct {name} for reusable infrastructure components"
        else:
            return f"CDK class {name} for AWS resource management"
    else:
        if 'lambda' in name.lower() or 'handler' in name.lower():
            return f"Lambda function handler {name}"
        elif services:
            return f"CDK helper function for {', '.join(services[:3])} operations"
        else:
            return f"CDK helper function {name}"

def extract_python_examples(content: str, file_path: str) -> List[Dict]:
    """Extract CDK constructs, stacks, and functions from Python code."""
    extracted = []

    try:
        tree = ast.parse(content)
        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Get the docstring
                docstring = ast.get_docstring(node)

                # Get the source code
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', len(lines))

                code = '\n'.join(lines[start_line:end_line])
                if len(code) < 80:
                    continue

                # If no docstring, try to extract preceding comment block
                if not docstring:
                    docstring = get_preceding_comment_block(
                        lines, start_line, lang='py', max_lines=20, min_chars=20
                    )

                # Generate meaningful docstrings for CDK constructs
                if not docstring:
                    docstring = generate_python_docstring(node, code)

                # Filter out low-quality comments
                if "TODO" in docstring or "fixme" in docstring.lower():
                    continue

                # More lenient filtering - just check for reasonable size
                if docstring and len(code.strip()) > 50:  # Lowered threshold
                    extracted.append({
                        'input': docstring.strip(),
                        'output': code.strip(),
                        'language': 'python',
                    })

    except SyntaxError:
        pass  # Skip files with syntax errors
    except Exception as e:
        print(f"Error processing Python file {file_path}: {e}")

    return extracted

def extract_typescript_examples(content: str, file_path: str) -> List[Dict]:
    """Extract CDK classes/functions from TypeScript code with preceding comment/JSDoc pairing."""
    extracted: List[Dict] = []
    lines = content.splitlines()

    TS_DECL_RE = re.compile(
        r"""
        (?P<export>\bexport\b\s+)?                                   # optional export
        (?:
            (?P<class>class)\s+(?P<class_name>[A-Za-z_]\w*)\s*        # class Name
            (?:extends\s+[A-Za-z0-9_.<>]+\s*)?                        # optional extends
            |
            (?P<function>function)\s+(?P<fn_name>[A-Za-z_]\w*)\s*     # function name
            |
            (?:const|let|var)\s+(?P<var_name>[A-Za-z_]\w*)\s*=\s*     # const name =
            (?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>\s*           # (...) =>  OR x =>
        )
        \{                                                            # opening brace of body
        """,
        re.VERBOSE | re.MULTILINE,
    )

    for m in TS_DECL_RE.finditer(content):
        # Identify declaration kind + name
        if m.group("class"):
            construct_type = "class"
            name = m.group("class_name")
        else:
            construct_type = "function"
            name = m.group("fn_name") or m.group("var_name")

        if not name:
            continue

        open_brace_idx = m.end() - 1  # points at '{'
        close_brace_idx = _find_matching_brace(content, open_brace_idx)
        if close_brace_idx is None:
            continue

        snippet = content[m.start(): close_brace_idx + 1].strip()
        if len(snippet) < 120:  # TS tends to be more verbose; keep a higher floor
            continue

        start_line = _offset_to_line_idx(content, m.start())
        docstring = get_preceding_comment_block(
            lines, start_line, lang="ts", max_lines=20, min_chars=20
        )

        if not docstring:
            docstring = generate_typescript_docstring(name, snippet, construct_type)

        extracted.append({
            "input": docstring.strip(),
            "output": snippet,
            "language": "typescript",
            #"meta": {
            #    "language": "typescript",
            #    "file_path": file_path,
            #    "construct_name": name,
            #    "construct_type": construct_type,
            #    "cdk_services": extract_cdk_services(snippet),
            #}
        })

    return extracted

def split_and_save_examples(
    examples: List[Dict],
    path: Path,
    seed: int = 42,
) -> None:
    if len(examples) > 0:
        print(f"Spliting Examples: {len(examples)} into Train/Validation")

        import random
        random.seed(seed)
        random.shuffle(examples)

        split_idx = int(0.9 * len(examples))
        print(f"Split Index at: {split_idx}")

        train_data = examples[:split_idx]
        val_data = examples[split_idx:]

        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        save_datasets(train_data, val_data, path)
    else:
        print("\nNo valid samples found after filtering. You may need to adjust the filtering criteria.")


def save_datasets(
        train_data: List[Dict],
        val_data: List[Dict],
        path: Path,
) -> None:

    with open(path / "train.jsonl", 'w') as f:
        for sample in train_data:
            f.write(json.dumps(sample) + '\n')

    with open(path / "validation.jsonl", 'w') as f:
        for sample in val_data:
            f.write(json.dumps(sample) + '\n')

    print("\nStage A dataset saved successfully!")
    print(f"Files saved to: {path}")

def load_train_val_data(path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load JSONL file into a list of dictionaries."""
    train_data = []
    val_data = []

    train_path = os.path.join(path, "train.jsonl")
    val_path = os.path.join(path, "validation.jsonl")

    with open(train_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    with open(val_path, 'r') as f:
        for line in f:
            val_data.append(json.loads(line))

    return train_data, val_data


# ============================================================================
# TS/JS Private Helpers
# ============================================================================

def _find_matching_brace(text: str, open_brace_idx: int) -> Optional[int]:
    """Return index of the matching '}' for the '{' at open_brace_idx, or None."""
    depth = 0
    in_str: Optional[str] = None
    esc = False
    i = open_brace_idx

    while i < len(text):
        ch = text[i]

        # string handling (basic, good enough for TS examples)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue

        if ch in ("'", '"', "`"):
            in_str = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return None


def _offset_to_line_idx(text: str, offset: int) -> int:
    """Convert a character offset into a 0-based line index."""
    return text.count("\n", 0, offset)