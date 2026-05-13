"""ahp.llm — chat-model helpers for adapters that need an LLM.

Currently only AWS Bedrock is wired up. Credentials are picked up from
the standard boto3 chain (``AWS_PROFILE``, env vars, IAM role, etc.) —
this module deliberately does NOT manage secrets.
"""

from ahp.llm.bedrock import (
    BEDROCK_DEFAULT_MODEL_ID,
    bedrock_chat_model,
    default_bedrock_model_id,
    has_aws_credentials,
)

__all__ = [
    "BEDROCK_DEFAULT_MODEL_ID",
    "bedrock_chat_model",
    "default_bedrock_model_id",
    "has_aws_credentials",
]
