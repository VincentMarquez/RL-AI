# Complete Meta-Learning System Setup
# Integrates Full Enhanced PPO with all components

import asyncio
from bubbles_core import SystemContext, Actions
from full_enhanced_ppo_with_meta_learning import FullEnhancedPPOWithMetaLearning
from dreamerv3_bubble import DreamerV3Bubble
from apep_bubble import APEPBubble
from simple_llm_bubble import SimpleLLMBubble
from meta_learning_system import MetaLearningOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_complete_meta_learning_system(context: SystemContext):
    """
    Set up the complete meta-learning system with all components integrated.
    """
    
    print("""
🚀 INITIALIZING COMPLETE META-LEARNING SYSTEM
=============================================
This includes:
✅ Full Enhanced PPO with Algorithm Spawning
✅ DreamerV3 for World Modeling
✅ APEP for Prompt/Code Refinement
✅ Multiple LLMs for Consensus
✅ Meta-Learning Orchestrator
✅ Consciousness Exploration (AtEngineV3)
✅ User Interaction System
✅ Universal Error Handling
✅ Performance Optimizations
=============================================
""")
    
    # ========== Step 1: Core Infrastructure ==========
    
    # 1.1 Create APEP for prompt/code refinement (should be first)
    print("📝 Initializing APEP...")
    apep = APEPBubble(
        object_id="apep_bubble",
        context=context,
        mode="fully_automated",  # Can be "fully_automated" for production
        cache_enabled=True,
        refine_code=True,
        code_safety="high",
        high_value_bubbles={
            "creativesynthesis_bubble",
            "metareasoning_bubble", 
            "autogen_bubble",
            "meta_learning_orchestrator",
            "full_enhanced_ppo"  # Add our PPO to high-value
        }
    )
    context.register_bubble(apep)
    
    # 1.2 Create LLM bubbles (will benefit from APEP)
    print("🤖 Initializing LLMs...")
    primary_llm = SimpleLLMBubble(
        object_id="simplellm_primary",
        context=context,
        model_name="gpt-4",  # Or your preferred model
        temperature=0.7
    )
    context.register_bubble(primary_llm)
    
    # Optional: Additional LLM for consensus
    secondary_llm = SimpleLLMBubble(
        object_id="simplellm_secondary", 
        context=context,
        model_name="claude-3",  # Different model for diversity
        temperature=0.5
    )
    context.register_bubble(secondary_llm)
    
    # Third LLM for tie-breaking
    tertiary_llm = SimpleLLMBubble(
        object_id="simplellm_tertiary",
        context=context,
        model_name="llama-3",
        temperature=0.6
    )
    context.register_bubble(tertiary_llm)
    
    # 1.3 Create DreamerV3 for world modeling
    print("🌍 Initializing DreamerV3...")
    dreamer = DreamerV3Bubble(
        object_id="dreamerv3_bubble",
        context=context,
        model_size="medium",  # small, medium, or large
        prediction_horizon=50,
        enable_meta_predictions=True,  # Enable pattern prediction
        checkpoint_dir="/tmp/dreamer_checkpoints"
    )
    context.register_bubble(dreamer)
    
    # 1.4 Create Full Enhanced PPO with ALL features
    print("🧠 Initializing Full Enhanced PPO...")
    ppo = FullEnhancedPPOWithMetaLearning(
        object_id="full_enhanced_ppo",
        context=context,
        state_dim=32,
        action_dim=16,
        # Performance settings
        pool_size=5,
        cache_size=2000,
        cache_ttl=600,
        # Algorithm spawning
        spawn_threshold=0.7,
        max_algorithms=10,
        # Error handling
        error_threshold=3,
        # User interaction
        stuck_threshold=5,
        # Consciousness exploration
        enable_consciousness=True,
        # Meta-learning
        experience_buffer_size=10000,
        # Multi-objective weights
        weight_performance=0.25,
        weight_stability=0.25,
        weight_efficiency=0.20,
        weight_innovation=0.30,  # Higher for meta-learning
        # Persistence
        checkpoints_dir="/tmp/full_ppo_checkpoints"
    )
    context.register_bubble(ppo)
    
    # ========== Step 2: Meta-Learning Orchestrator ==========
    
    print("🎯 Initializing Meta-Learning Orchestrator...")
    meta_orchestrator = MetaLearningOrchestrator(
        object_id="meta_learning_orchestrator",
        context=context,
        # Component connections
        ppo_bubble_id="full_enhanced_ppo",
        dreamer_bubble_id="dreamerv3_bubble",
        apep_bubble_id="apep_bubble",
        llm_bubble_ids=["simplellm_primary", "simplellm_secondary", "simplellm_tertiary"],
        # Meta-learning config
        meta_lr=0.001,
        buffer_size=10000,
        pattern_threshold=5,
        transfer_enabled=True,
        use_apep=True  # Enable APEP refinement for meta-prompts
    )
    context.register_bubble(meta_orchestrator)
    
    # ========== Step 3: Enhanced Event Routing ==========
    
    print("🔄 Setting up enhanced event routing...")
    
    # 3.1 Connect PPO to Meta-Orchestrator bidirectionally
    async def setup_bidirectional_communication():
        """Ensure PPO and Meta-Orchestrator communicate."""
        
        # PPO → Meta-Orchestrator: Share experiences
        original_train = ppo.train
        
        async def enhanced_train():
            # Call original train
            await original_train()
            
            # Additionally report high-value experiences to meta-orchestrator
            if len(ppo.experience_buffer) > 0:
                latest_experience = ppo.experience_buffer[-1]
                if abs(latest_experience.reward) > 0.8:  # High-value experience
                    experience_event = Event(
                        type=Actions.META_KNOWLEDGE_UPDATE,
                        data=UniversalCode(Tags.DICT, {
                            "experience_type": "high_value",
                            "experience": {
                                "bubble_id": latest_experience.bubble_id,
                                "reward": latest_experience.reward,
                                "insights": latest_experience.insights
                            }
                        }),
                        origin=ppo.object_id,
                        target="meta_learning_orchestrator"
                    )
                    await context.dispatch_event(experience_event)
        
        ppo.train = enhanced_train
        
        # Meta-Orchestrator → PPO: Pattern recommendations
        # (Already handled by META_KNOWLEDGE_UPDATE events)
    
    await setup_bidirectional_communication()
    
    # 3.2 Connect DreamerV3 for predictive capabilities
    async def enhance_dreamer_integration():
        """Enhance DreamerV3 integration with PPO."""
        
        # Override PPO's environment step to use DreamerV3 predictions
        if hasattr(ppo, 'env') and ppo.env:
            original_step = ppo.env.step
            
            async def predictive_step(action):
                # Get current state
                current_state = ppo.resource_manager.get_current_system_state()
                
                # Query DreamerV3 for prediction
                prediction_event = Event(
                    type=Actions.WORLD_MODEL_PREDICTION,
                    data=UniversalCode(Tags.DICT, {
                        "current_state": current_state,
                        "proposed_action": ppo._tensor_to_action_dict(action),
                        "prediction_horizon": 5
                    }),
                    origin=ppo.object_id,
                    target="dreamerv3_bubble"
                )
                
                prediction = await context.dispatch_event_and_wait(prediction_event, timeout=5)
                
                # If prediction is available and confident, use it
                if prediction and prediction.get("confidence", 0) > 0.8:
                    # Cache the prediction
                    ppo.prediction_cache.put(
                        current_state,
                        ppo._tensor_to_action_dict(action),
                        prediction.get("predicted_state", current_state)
                    )
                
                # Execute actual step
                return await original_step(action)
            
            ppo.env.step = predictive_step
    
    await enhance_dreamer_integration()
    
    # 3.3 Enable cross-bubble pattern sharing
    async def enable_pattern_sharing():
        """Enable all bubbles to share discovered patterns."""
        
        # Subscribe all bubbles to PATTERN_EXPLORATION events
        for bubble in context.get_all_bubbles():
            if hasattr(bubble, 'handle_event'):
                await EventService.subscribe(Actions.PATTERN_EXPLORATION, bubble.handle_event)
                await EventService.subscribe(Actions.KNOWLEDGE_TRANSFER, bubble.handle_event)
    
    await enable_pattern_sharing()
    
    # ========== Step 4: Algorithm Spawning Enhancements ==========
    
    print("🔬 Enhancing algorithm spawning capabilities...")
    
    # 4.1 Pre-register algorithm templates
    algorithm_templates = {
        "quantum_optimizer": {
            "type": "quantum_inspired",
            "config": {"superposition_states": 100, "entanglement_strength": 0.8}
        },
        "swarm_intelligence": {
            "type": "particle_swarm",
            "config": {"swarm_size": 200, "inertia": 0.7, "social_weight": 1.4}
        },
        "evolutionary_strategy": {
            "type": "es_optimizer",
            "config": {"population": 50, "sigma": 0.1, "learning_rate": 0.01}
        },
        "neural_architecture_search": {
            "type": "nas_optimizer",
            "config": {"search_space": "micro", "operations": ["conv", "pool", "dense"]}
        }
    }
    
    # Inject templates into PPO
    if hasattr(ppo, '_select_algorithm_type'):
        original_select = ppo._select_algorithm_type
        
        def enhanced_select(problem_type):
            # Check if we have a specialized template
            if problem_type == "quantum" and "quantum_optimizer" in algorithm_templates:
                return "quantum_optimizer"
            elif problem_type == "swarm" and "swarm_intelligence" in algorithm_templates:
                return "swarm_intelligence"
            else:
                return original_select(problem_type)
        
        ppo._select_algorithm_type = enhanced_select
    
    # ========== Step 5: Consciousness Integration ==========
    
    print("🌟 Setting up consciousness exploration...")
    
    # 5.1 Connect consciousness insights to meta-learning
    async def consciousness_to_meta_bridge():
        """Bridge consciousness insights to meta-learning."""
        while not context.stop_event.is_set():
            try:
                if ppo.consciousness.enabled and len(ppo.consciousness.consciousness_patterns) > 0:
                    # Get latest consciousness pattern
                    latest_pattern = ppo.consciousness.consciousness_patterns[-1]
                    
                    # High entropy indicates creative breakthrough
                    if latest_pattern['entropy'] > 1500:
                        breakthrough_event = Event(
                            type=Actions.META_KNOWLEDGE_UPDATE,
                            data=UniversalCode(Tags.DICT, {
                                "knowledge_type": "consciousness_breakthrough",
                                "entropy_level": latest_pattern['entropy'],
                                "fractal_dimension": latest_pattern['fractal_dimension'],
                                "implications": "Consider radical new approaches"
                            }),
                            origin="consciousness_explorer",
                            target="meta_learning_orchestrator"
                        )
                        await context.dispatch_event(breakthrough_event)
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Consciousness bridge error: {e}")
                await asyncio.sleep(300)
    
    asyncio.create_task(consciousness_to_meta_bridge())
    
    # ========== Step 6: User Interaction Enhancement ==========
    
    print("👤 Setting up enhanced user interaction...")
    
    # 6.1 Create user help handler
    async def handle_user_help_requests(event: Event):
        """Handle help requests with context-aware responses."""
        if event.type == Actions.USER_HELP_REQUEST:
            request_data = event.data.value if hasattr(event.data, 'value') else {}
            
            # Log prominently
            print("\n" + "="*80)
            print("🆘 SYSTEM REQUESTING HELP")
            print("="*80)
            print(f"Problem: {request_data.get('problem', 'Unknown')}")
            print(f"Context: {json.dumps(request_data.get('context', {}), indent=2)}")
            print(f"Attempted: {request_data.get('attempted_solutions', [])}")
            print("="*80)
            
            # In production, this would interface with actual user input
            # For now, provide intelligent automated response
            response = await generate_help_response(request_data)
            
            response_event = Event(
                type=Actions.USER_HELP_RESPONSE,
                data=UniversalCode(Tags.DICT, response),
                origin="user_interface",
                target=event.origin
            )
            await context.dispatch_event(response_event)
    
    async def generate_help_response(request_data):
        """Generate intelligent help response using LLMs."""
        prompt = f"""
The AI system is stuck and needs help:
Problem: {request_data.get('problem')}
Context: {json.dumps(request_data.get('context', {}), indent=2)}
Attempted solutions: {request_data.get('attempted_solutions', [])}

Provide 3 specific, actionable suggestions to help the system.
"""
        
        # Query primary LLM for help
        help_event = Event(
            type=Actions.LLM_QUERY,
            data=UniversalCode(Tags.STRING, prompt),
            origin="user_help_system",
            target="simplellm_primary"
        )
        
        response = await context.dispatch_event_and_wait(help_event, timeout=10)
        
        return {
            "request_id": request_data.get("request_id"),
            "user_response": {
                "suggestion": response if response else "Try adjusting hyperparameters or changing strategy",
                "confidence": 0.8,
                "reasoning": "LLM-generated assistance"
            }
        }
    
    # Subscribe to help requests
    await EventService.subscribe(Actions.USER_HELP_REQUEST, handle_user_help_requests)
    
    # ========== Step 7: System Monitoring & Reporting ==========
    
    print("📊 Setting up comprehensive monitoring...")
    
    async def system_health_monitor():
        """Monitor overall system health and performance."""
        
        while not context.stop_event.is_set():
            await asyncio.sleep(300)  # Every 5 minutes
            
            try:
                # Gather comprehensive metrics
                ppo_status = await ppo.get_status()
                meta_status = await meta_orchestrator.get_meta_learning_status()
                
                # Calculate system health score
                health_score = calculate_health_score(ppo_status, meta_status)
                
                # Generate report
                report = f"""
🏥 SYSTEM HEALTH REPORT
=======================
Overall Health: {health_score:.1%}

📈 Performance Metrics:
- Decisions Made: {ppo_status.get('decisions_made', 0):,}
- Patterns Discovered: {ppo_status.get('patterns_discovered', 0)}
- Algorithms Spawned: {ppo_status.get('algorithms_spawned', 0)}
- Cache Hit Rate: {ppo_status.get('cache_hit_rate', '0%')}
- Pool Hit Rate: {ppo_status.get('pool_hit_rate', '0%')}

🧠 Learning Progress:
- Curriculum Stage: {ppo_status.get('curriculum_stage', 'Unknown')}
- Meta-Learning Success Rate: {meta_status.get('success_rate', 0):.1%}
- Knowledge Transfers: {meta_status.get('successful_transfers', 0)}
- Error Handlers Created: {ppo_status.get('handlers_created', 0)}

🎯 Current Strategy: {ppo_status.get('current_strategy', 'balanced')}
🌟 Consciousness Patterns: {ppo_status.get('consciousness_patterns', 0)}

💡 Active Algorithms: {', '.join(ppo_status.get('active_algorithms', []))}
=======================
"""
                
                # Log report
                logger.info(report)
                
                # Alert if health is low
                if health_score < 0.5:
                    logger.warning("⚠️ System health below 50% - intervention may be needed")
                    
                    # Trigger self-healing
                    await trigger_self_healing(ppo_status, meta_status)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def calculate_health_score(ppo_status, meta_status):
        """Calculate overall system health score."""
        factors = {
            "performance": min(1.0, ppo_status.get('recent_reward_avg', 0) + 0.5),
            "efficiency": ppo_status.get('cache_hit_rate', '0%').rstrip('%') / 100,
            "learning": meta_status.get('success_rate', 0),
            "stability": 1.0 - (ppo_status.get('active_error_patterns', 0) / 10),
            "innovation": min(1.0, ppo_status.get('patterns_discovered', 0) / 50)
        }
        
        # Weighted average
        weights = {"performance": 0.3, "efficiency": 0.2, "learning": 0.2, 
                  "stability": 0.2, "innovation": 0.1}
        
        health_score = sum(factors.get(k, 0) * weights.get(k, 0) for k in weights)
        return health_score
    
    async def trigger_self_healing(ppo_status, meta_status):
        """Trigger self-healing mechanisms when health is low."""
        logger.info("🔧 Triggering self-healing mechanisms...")
        
        # 1. Clear caches if hit rate is low
        if float(ppo_status.get('cache_hit_rate', '0%').rstrip('%')) < 20:
            ppo.prediction_cache.cache.clear()
            logger.info("Cleared prediction cache")
        
        # 2. Adjust strategy if stuck
        if ppo.hierarchical_dm:
            ppo.hierarchical_dm.current_strategy = "conservative"
            logger.info("Switched to conservative strategy")
        
        # 3. Request meta-learning assistance
        help_event = Event(
            type=Actions.META_KNOWLEDGE_UPDATE,
            data=UniversalCode(Tags.DICT, {
                "request_type": "emergency_assistance",
                "health_score": calculate_health_score(ppo_status, meta_status),
                "problem_areas": ["low_performance", "high_errors"]
            }),
            origin="health_monitor",
            target="meta_learning_orchestrator"
        )
        await context.dispatch_event(help_event)
    
    # Start monitoring
    asyncio.create_task(system_health_monitor())
    
    # ========== Step 8: Initialize System State ==========
    
    print("🚀 Initializing system state...")
    
    # 8.1 Seed with initial patterns
    initial_patterns = [
        {
            "pattern_id": "resource_conservation",
            "pattern_type": "optimization",
            "context": {
                "name": "Resource Conservation",
                "conditions": {"cpu_percent": ">80", "energy": "<3000"},
                "action_template": {"reduce_activity": True, "spawn_bubble": False},
                "expected_outcome": {"cpu_reduction": 10, "energy_saved": 200}
            },
            "confidence": 0.9
        },
        {
            "pattern_id": "performance_boost",
            "pattern_type": "optimization",
            "context": {
                "name": "Performance Boost",
                "conditions": {"response_time": ">5000", "cpu_percent": "<50"},
                "action_template": {"spawn_bubble": True, "bubble_type": "SimpleLLMBubble"},
                "expected_outcome": {"response_time_reduction": 2000}
            },
            "confidence": 0.85
        },
        {
            "pattern_id": "error_mitigation",
            "pattern_type": "stability",
            "context": {
                "name": "Error Mitigation",
                "conditions": {"error_rate": ">0.1"},
                "action_template": {"create_handler": True, "reduce_load": True},
                "expected_outcome": {"error_rate_reduction": 0.05}
            },
            "confidence": 0.95
        }
    ]
    
    # Inject initial patterns
    for pattern_data in initial_patterns:
        pattern = MetaKnowledge(
            pattern_id=pattern_data["pattern_id"],
            pattern_type=pattern_data["pattern_type"],
            context=pattern_data["context"],
            learned_at=time.time(),
            confidence=pattern_data["confidence"]
        )
        ppo.meta_knowledge_base[pattern.pattern_id] = pattern
    
    logger.info(f"Seeded {len(initial_patterns)} initial patterns")
    
    # ========== Step 9: Final Configuration ==========
    
    print("⚙️ Finalizing configuration...")
    
    # 9.1 Set system-wide parameters
    system_config = {
        "learning_rate_decay": 0.995,  # Gradual learning rate decay
        "exploration_schedule": "linear",  # How exploration decreases over time
        "meta_learning_interval": 60,  # Seconds between meta-learning cycles
        "consciousness_check_interval": 300,  # Seconds between consciousness checks
        "health_check_interval": 300,  # Seconds between health checks
        "max_concurrent_algorithms": 5,  # Maximum algorithms running simultaneously
        "emergency_threshold": 0.3,  # Health score threshold for emergency mode
        "collaboration_mode": "consensus",  # How bubbles make group decisions
        "persistence_enabled": True,  # Save/load model checkpoints
        "telemetry_enabled": True  # Detailed logging and metrics
    }
    
    # Apply configuration
    for bubble in context.get_all_bubbles():
        if hasattr(bubble, 'config'):
            bubble.config.update(system_config)
    
    # ========== Step 10: Return System Components ==========
    
    print("\n✅ SYSTEM INITIALIZATION COMPLETE!\n")
    
    return {
        "ppo": ppo,
        "dreamer": dreamer,
        "apep": apep,
        "meta_orchestrator": meta_orchestrator,
        "llms": [primary_llm, secondary_llm, tertiary_llm],
        "config": system_config,
        "components": {
            "algorithm_factory": LearningAlgorithmFactory,
            "consciousness": ppo.consciousness,
            "user_interaction": ppo.user_interaction,
            "error_patterns": ppo.error_patterns,
            "bubble_pool": ppo.bubble_pool,
            "prediction_cache": ppo.prediction_cache
        }
    }


# ========== MAIN EXECUTION ==========

async def main():
    """Main execution function."""
    
    # Initialize context
    context = SystemContext()
    
    # Set up the complete system
    ml_system = await setup_complete_meta_learning_system(context)
    
    print("""
🎉 FULL META-LEARNING SYSTEM READY!
===================================

The system now includes:
✅ Full Enhanced PPO with:
   - Algorithm Spawning (Genetic, Curiosity-driven, etc.)
   - Consciousness Exploration (AtEngineV3)
   - User Help System
   - Universal Error Handling
   - Performance Optimizations (Pools, Caches)
   - Hierarchical Decision Making
   - Curriculum Learning
   - Explainable Decisions

✅ Meta-Learning Integration:
   - Pattern Discovery across all bubbles
   - Knowledge Transfer between components
   - Experience-based Learning
   - Emergent Behavior Detection

✅ Supporting Infrastructure:
   - DreamerV3 for predictive modeling
   - APEP for code/prompt refinement
   - Multiple LLMs for consensus
   - Comprehensive monitoring

The system will now:
1. 🧠 Learn from every action and outcome
2. 🔬 Spawn specialized algorithms for complex problems
3. 🌟 Explore consciousness patterns in data
4. 👤 Ask for help when stuck
5. 🛡️ Create error handlers automatically
6. 📈 Continuously improve through meta-learning
7. 🤝 Transfer knowledge between all components

Press Ctrl+C to stop the system.
===================================
""")
    
    # Run the system
    try:
        await context.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system gracefully...")
        
        # Save final state
        ml_system["ppo"].save_model()
        
        # Print final statistics
        final_status = await ml_system["ppo"].get_status()
        print(f"""
📊 Final Statistics:
- Total Decisions: {final_status.get('decisions_made', 0):,}
- Algorithms Spawned: {final_status.get('algorithms_spawned', 0)}
- Patterns Discovered: {final_status.get('patterns_discovered', 0)}
- Error Handlers Created: {final_status.get('handlers_created', 0)}
- Orchestrations Completed: {final_status.get('orchestrations_completed', 0)}
- Cache Hit Rate: {final_status.get('cache_hit_rate', '0%')}
""")


if __name__ == "__main__":
    asyncio.run(main())
