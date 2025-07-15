from PPOBubble import FullEnhancedPPOWithMetaLearning
from complete_meta_orchestrator_final import MetaLearningOrchestrator
import logging

logger = logging.getLogger(__name__)

# Alias for compatibility
FullyIntegratedPPO = FullEnhancedPPOWithMetaLearning

async def setup_full_ppo_with_meta_learning(context, **kwargs):
    """Setup Full Enhanced PPO with Meta-Learning."""
    ppo_id = kwargs.get('ppo_id', 'enhanced_ppo')
    
    # Create the PPO instance
    ppo = FullEnhancedPPOWithMetaLearning(
        object_id=ppo_id,
        context=context,
        state_dim=kwargs.get('state_dim', 24),
        action_dim=kwargs.get('action_dim', 5),
        pool_size=kwargs.get('pool_size', 3),
        cache_size=kwargs.get('cache_size', 1000),
        cache_ttl=kwargs.get('cache_ttl', 300),
        spawn_threshold=kwargs.get('spawn_threshold', 0.7),
        max_algorithms=kwargs.get('max_algorithms', 10),
        error_threshold=kwargs.get('error_threshold', 3),
        weight_performance=kwargs.get('weight_performance', 0.3),
        weight_stability=kwargs.get('weight_stability', 0.3),
        weight_efficiency=kwargs.get('weight_efficiency', 0.2),
        weight_innovation=kwargs.get('weight_innovation', 0.2)
    )
    
    result = {"ppo": ppo}
    
    # Optionally create meta orchestrator
    if kwargs.get('use_meta_orchestrator', True):
        meta_orchestrator = MetaLearningOrchestrator(
            object_id=f"{ppo_id}_meta_orchestrator",
            context=context,
            ppo_bubble_id=ppo_id,
            dreamer_bubble_id=kwargs.get('dreamer_bubble_id', 'dreamer_bubble'),
            llm_bubble_ids=kwargs.get('llm_bubble_ids', ['simplellm_bubble'])
        )
        result["meta_orchestrator"] = meta_orchestrator
    
    return result
