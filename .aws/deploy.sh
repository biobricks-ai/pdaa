# Rebuild the docker image and push it to ECR
# Destroy the service 
# Destroy the instances in the capacity provider
# Create a new service with the updated image
export AWS_REGION="us-east-1"
export AWS_PROFILE=chemprop-transformer
aws sso login --profile $AWS_PROFILE
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

# Load secrets =================================================================
echo "Reading chemprop-transformer secret..." >&2
SECRET_NAME="github.com/biobricks-ai/pdaa"
SECRET=$(aws secretsmanager get-secret-value --secret-id $SECRET_NAME --query 'SecretString' --output text)

# Variables (update these for your AWS setup)
export ECR_REPO_NAME='biobricks-ai/pdaa'
export DOCKER_IMAGE_TAG='latest'
export INSTANCE_TYPE='r6g.xlarge'  # 32 GB RAM instance type for ECS
export AMI_ID='ami-04b9e92b5572fa0d1'  # Amazon ECS-optimized AMI with GPU support
export KEY_NAME='pdaa-key'  # SSH key pair name

# Build and push Docker image to ECR ===========================================
# depends on: [ ]
echo "Building Docker image..."
docker build -t $ECR_REPO_NAME .

echo "Tagging Docker image for ECR..."
docker tag "$ECR_REPO_NAME:$DOCKER_IMAGE_TAG" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$DOCKER_IMAGE_TAG"

echo "Pushing Docker image to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr describe-repositories --repository-names $ECR_REPO_NAME >/dev/null 2>&1 || aws ecr create-repository --repository-name $ECR_REPO_NAME
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$DOCKER_IMAGE_TAG

# start ec2 instance
aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --subnet-id $SUBNET_ID \
    --iam-instance-profile Name=$IAM_INSTANCE_PROFILE \
    --launch-template Name=$LAUNCH_TEMPLATE_NAME \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=chemprop-transformer}]"
