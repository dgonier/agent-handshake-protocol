"""ahp.llm — chat-model helpers and recipe primitives.

Three natural model / compute sources for AHP:

* **Bedrock** (:mod:`ahp.llm.bedrock`) — hosted Claude / Llama / Titan
  / Mistral on AWS.
* **OpenRouter** (:mod:`ahp.llm.openrouter`) — OpenAI-compatible
  unified API in front of 100+ models, useful for A/B-testing
  recipes across providers.
* **Modal** — serverless GPUs you bring to AHP either as
  OpenAI-compatible endpoints (use the OpenRouter helper pointed at
  ``base_url=https://you--app.modal.run``) or as full AHP nodes that
  host their own agents. No library code needed — see
  ``examples/modal_*`` (planned).

Plus the recipe primitives in :mod:`ahp.llm.recipe`:

* ``ModelHandle`` / ``LoRAHandle`` — pure-metadata handles to base
  models and adapters, registered via the standard
  ``@resource("scope", "model"|"lora", ...)`` decorator.
* ``find_model`` / ``find_loras`` / ``recipe_summary`` — inspect an
  :class:`AgentProfile` to assemble the recipe.

Credentials are NEVER read directly by these modules — boto3 and
``OPENROUTER_API_KEY`` handle that. Each module exposes a
``has_…_credentials()`` pre-flight so tests / demos can skip
gracefully when running outside the relevant cloud.
"""

from ahp.llm.bedrock import (
    BEDROCK_DEFAULT_MODEL_ID,
    bedrock_chat_model,
    default_bedrock_model_id,
    has_aws_credentials,
)
from ahp.llm.openrouter import (
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_DEFAULT_MODEL_ID,
    default_openrouter_model_id,
    has_openrouter_credentials,
    openrouter_chat_model,
)
from ahp.llm.recipe import (
    LoRAHandle,
    ModelHandle,
    all_recipe_handles,
    find_loras,
    find_model,
    recipe_summary,
)

__all__ = [
    # Bedrock
    "BEDROCK_DEFAULT_MODEL_ID",
    "bedrock_chat_model",
    "default_bedrock_model_id",
    "has_aws_credentials",
    # OpenRouter
    "OPENROUTER_DEFAULT_BASE_URL",
    "OPENROUTER_DEFAULT_MODEL_ID",
    "default_openrouter_model_id",
    "has_openrouter_credentials",
    "openrouter_chat_model",
    # Recipe primitives
    "LoRAHandle",
    "ModelHandle",
    "all_recipe_handles",
    "find_loras",
    "find_model",
    "recipe_summary",
]
