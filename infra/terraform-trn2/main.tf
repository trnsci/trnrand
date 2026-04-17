terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# trn2.3xlarge is available in sa-east-1 (a, b, c) as of 2026-04-16.
# trn2.xlarge does not exist; trn2.3xlarge is the smallest trn2 instance.
variable "aws_region" {
  description = "AWS region for the CI instance (trn2.3xlarge available in sa-east-1)"
  type        = string
  default     = "sa-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "trn2.3xlarge"
}

variable "instance_tag" {
  description = "Name tag used by run_neuron_tests.sh to find the instance"
  type        = string
  default     = "trnrand-ci-trn2"
}

variable "az_suffix" {
  description = "AZ suffix for the subnet (a, b, or c). Change if InsufficientInstanceCapacity."
  type        = string
  default     = "a"
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Deep Learning AMI with Neuron SDK pre-installed
# ---------------------------------------------------------------------------

data "aws_ami" "neuron" {
  most_recent = true
  owners      = ["amazon"]

  # Any PyTorch-2.x Neuron AMI on Ubuntu 24.04. most_recent=true picks the
  # newest, which (as of Neuron SDK 2.29, April 2026) bundles
  # neuronxcc 2.29 / NKI 0.3.0. Widening the filter rather than pinning
  # a specific PyTorch minor lets AMI releases land without a TF edit.
  filter {
    name   = "name"
    values = ["Deep Learning AMI Neuron PyTorch 2.*Ubuntu 24.04*"]
  }
}

# ---------------------------------------------------------------------------
# VPC + public subnet (sa-east-1 has no pre-existing VPC for this CI account)
# ---------------------------------------------------------------------------

resource "aws_vpc" "ci" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "${var.instance_tag}-vpc" }
}

resource "aws_internet_gateway" "ci" {
  vpc_id = aws_vpc.ci.id
  tags   = { Name = "${var.instance_tag}-igw" }
}

resource "aws_subnet" "ci" {
  vpc_id                  = aws_vpc.ci.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}${var.az_suffix}"
  map_public_ip_on_launch = true
  tags = { Name = "${var.instance_tag}-subnet" }
}

resource "aws_route_table" "ci" {
  vpc_id = aws_vpc.ci.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ci.id
  }

  tags = { Name = "${var.instance_tag}-rt" }
}

resource "aws_route_table_association" "ci" {
  subnet_id      = aws_subnet.ci.id
  route_table_id = aws_route_table.ci.id
}

# ---------------------------------------------------------------------------
# IAM role for the EC2 instance (SSM access)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "instance" {
  name = "${var.instance_tag}-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.instance_tag}-profile"
  role = aws_iam_role.instance.name
}

# ---------------------------------------------------------------------------
# Security group (SSM only, no inbound)
# ---------------------------------------------------------------------------

resource "aws_security_group" "instance" {
  name        = "${var.instance_tag}-sg"
  description = "SSM-only access for trnrand CI (trn2)"
  vpc_id      = aws_vpc.ci.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# EC2 instance
# ---------------------------------------------------------------------------

resource "aws_instance" "ci" {
  ami                         = data.aws_ami.neuron.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.ci.id
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  vpc_security_group_ids      = [aws_security_group.instance.id]
  associate_public_ip_address = true # Needed for SSM agent to reach regional endpoint

  root_block_device {
    volume_size = 200 # trn2 NEFF caches grow larger; extra headroom vs trn1's 100 GB
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    cd /home/ubuntu
    sudo -u ubuntu git clone https://github.com/trnsci/trnrand.git trnrand
    # Install into the AMI's pre-built Neuron venv (has neuronxcc preinstalled).
    # Use [dev] only — [neuron] would try to fetch neuronxcc from PyPI where it doesn't exist.
    NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1)
    sudo -u ubuntu $NEURON_VENV/bin/pip install -e '/home/ubuntu/trnrand[dev]'
    # neuronxcc compile workdirs can exceed /tmp (tmpfs, RAM-backed).
    # Redirect to /var/tmp (EBS-backed, 200 GB) for all ubuntu sessions.
    echo 'export TMPDIR=/var/tmp' >> /home/ubuntu/.profile
  EOF

  tags = {
    Name = var.instance_tag
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_id" {
  value = aws_instance.ci.id
}

output "instance_tag" {
  value       = var.instance_tag
  description = "Name tag used by scripts/run_neuron_tests.sh"
}

output "aws_region" {
  value       = var.aws_region
  description = "Pass to run script: AWS_REGION=$(terraform output -raw aws_region)"
}

output "ami_id" {
  value       = data.aws_ami.neuron.id
  description = "Neuron Deep Learning AMI resolved at apply time"
}

output "vpc_id" {
  value       = aws_vpc.ci.id
  description = "VPC created for the CI instance"
}

output "subnet_id" {
  value       = aws_subnet.ci.id
  description = "Public subnet created for the CI instance"
}
