import * as cdk from 'aws-cdk-lib';
import { Duration, Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';

import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';

export class ApiStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // CloudWatch Log Group (explicit, with retention & removal policy)

    // IAM role for Lambda

    // Basic logging permissions (CloudWatch Logs)

    // Lambda function (Python 3.11)

  }