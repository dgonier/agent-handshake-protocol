# Single-node Neo4j 5 on an EC2 instance, behind a security group that
# only exposes Bolt + browser to the operator's CIDR. Suitable for an
# AHP development KG; for production use Neo4j AuraDB or a 3-node
# Causal Cluster instead.
#
# Apply::
#
#     terraform init
#     terraform apply \
#       -var="key_name=$YOUR_AWS_KEY" \
#       -var="allowed_cidr=$YOUR_IP/32" \
#       -var="neo4j_password=$(openssl rand -base64 24)"
#
# Outputs the bolt URL + browser URL. The vector index bootstrap runs
# automatically via user_data; SSH in and check /var/log/cloud-init-output.log
# if anything looks wrong.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── networking ────────────────────────────────────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "neo4j" {
  name        = "ahp-neo4j-${var.environment}"
  description = "AHP knowledge graph — Bolt + browser, locked to operator CIDR"
  vpc_id      = data.aws_vpc.default.id

  # Bolt (driver protocol)
  ingress {
    from_port   = 7687
    to_port     = 7687
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    description = "Neo4j Bolt"
  }

  # HTTP browser
  ingress {
    from_port   = 7474
    to_port     = 7474
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    description = "Neo4j browser"
  }

  # SSH for ops
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    description = "SSH"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "ahp-neo4j-${var.environment}"
    Project     = "ahp"
    Environment = var.environment
  }
}

# ── data + AMI ────────────────────────────────────────────────────────

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── instance ──────────────────────────────────────────────────────────

resource "aws_instance" "neo4j" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = element(data.aws_subnets.default.ids, 0)
  vpc_security_group_ids = [aws_security_group.neo4j.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.disk_gb
    delete_on_termination = true
    encrypted             = true
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    neo4j_password    = var.neo4j_password
    vector_dimensions = var.vector_dimensions
  })

  # Force re-create when the bootstrap inputs change.
  user_data_replace_on_change = true

  tags = {
    Name        = "ahp-neo4j-${var.environment}"
    Project     = "ahp"
    Environment = var.environment
  }
}
