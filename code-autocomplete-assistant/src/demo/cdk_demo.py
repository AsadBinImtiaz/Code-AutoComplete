from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_kms as kms,
    RemovalPolicy
)
from constructs import Construct


class DataStorageStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Create an S3 bucket with:
        # - Encryption enabled (KMS)
        # - Versioning enabled
        # - Block public access
        # - Lifecycle rule to transition to Glacier after 90 days

        # <CURSOR HERE - generate bucket code>

        # Expected:
        # encryption_key = kms.Key(self, "BucketKey",
        #     enable_key_rotation=True
        # )
        #
        # bucket = s3.Bucket(self, "DataBucket",
        #     encryption=s3.BucketEncryption.KMS,
        #     encryption_key=encryption_key,
        #     versioned=True,
        #     block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        #     lifecycle_rules=[
        #         s3.LifecycleRule(
        #             transitions=[
        #                 s3.Transition(
        #                     storage_class=s3.StorageClass.GLACIER,
        #                     transition_after=Duration.days(90)
        #                 )
        #             ]
        #         )
        #     ],
        #     removal_policy=RemovalPolicy.RETAIN
        # )