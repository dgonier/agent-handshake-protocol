variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region the Neo4j instance lives in."
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Free-form environment tag (dev/stage/prod). Becomes part of resource names."
}

variable "instance_type" {
  type        = string
  default     = "t3.medium"
  description = "EC2 instance shape. t3.medium = 2 vCPU / 4 GB; bump to m5.large for serious graphs."
}

variable "disk_gb" {
  type        = number
  default     = 100
  description = "Root EBS volume size in GB. Neo4j data + page cache live here."
}

variable "key_name" {
  type        = string
  description = "Existing EC2 key pair name. Required for SSH access."
}

variable "allowed_cidr" {
  type        = string
  description = "Source CIDR that may reach the instance (SSH + Bolt + browser). Use \"$(curl -s ifconfig.me)/32\"."
}

variable "neo4j_password" {
  type        = string
  description = "Password for the `neo4j` user. Pass via -var or TF_VAR_neo4j_password — don't commit."
  sensitive   = true
}

variable "vector_dimensions" {
  type        = number
  default     = 1536
  description = "Embedding dimensions baked into the vector index. Must match the embedder used by the writer agent."
}
