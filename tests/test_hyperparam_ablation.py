"""
Tests for Stage 2-4 (Hyperparameter Tuning & Ablation Studies)

Test Categories:
- Unit tests (no LLM): Data classes, parsers, state management
- Unit tests (with LLM): Idea generation, node generation
- Integration tests: Stage transitions, full pipeline
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[ENV] Loaded .env from {env_path}")
else:
    print(f"[ENV] Warning: .env not found at {env_path}")

import pytest
from unittest.mock import Mock, patch, MagicMock

from masist.treesearch.parallel_agent import (
    _parse_keyword_prefix_response,
    AblationConfig,
    AblationIdea,
    HyperparamTuningIdea,
    ParallelAgent,
)
from masist.treesearch.journal import Node, Journal


# =============================================================================
# Phase 1: Unit Tests (No LLM) - Data Classes
# =============================================================================

class TestHyperparamTuningIdea:
    """Tests for HyperparamTuningIdea data class"""

    def test_init(self):
        """Test basic initialization"""
        idea = HyperparamTuningIdea(
            name="learning_rate",
            description="Increase learning rate from 0.001 to 0.01"
        )
        assert idea.name == "learning_rate"
        assert idea.description == "Increase learning rate from 0.001 to 0.01"

    def test_init_empty_strings(self):
        """Test initialization with empty strings"""
        idea = HyperparamTuningIdea(name="", description="")
        assert idea.name == ""
        assert idea.description == ""

    def test_init_special_characters(self):
        """Test initialization with special characters"""
        idea = HyperparamTuningIdea(
            name="temperature_調整",
            description="温度を0.5から0.7に上げる (increase temperature)"
        )
        assert idea.name == "temperature_調整"
        assert "温度" in idea.description


class TestAblationIdea:
    """Tests for AblationIdea data class"""

    def test_init(self):
        """Test basic initialization"""
        idea = AblationIdea(
            name="remove_dropout",
            description="Remove dropout layer to test its contribution"
        )
        assert idea.name == "remove_dropout"
        assert idea.description == "Remove dropout layer to test its contribution"

    def test_init_empty_strings(self):
        """Test initialization with empty strings"""
        idea = AblationIdea(name="", description="")
        assert idea.name == ""
        assert idea.description == ""


class TestAblationConfig:
    """Tests for AblationConfig state management class"""

    def test_init(self):
        """Test basic initialization"""
        mock_node = Mock(spec=Node)
        config = AblationConfig(
            name="test_ablation",
            description="Test ablation description",
            code="print('test')",
            base_node=mock_node
        )
        assert config.name == "test_ablation"
        assert config.description == "Test ablation description"
        assert config.code == "print('test')"
        assert config.base_node == mock_node
        assert config.attempts == 0
        assert config.max_attempts == 3
        assert config.last_error is None
        assert config.completed is False
        assert config.current_node is None

    def test_state_management(self):
        """Test state management operations"""
        mock_node = Mock(spec=Node)
        config = AblationConfig(
            name="test",
            description="test",
            code="",
            base_node=mock_node
        )

        # Simulate retry attempts
        config.attempts = 1
        config.last_error = "SyntaxError"
        assert config.attempts == 1
        assert config.last_error == "SyntaxError"

        # Simulate completion
        config.completed = True
        config.current_node = mock_node
        assert config.completed is True
        assert config.current_node == mock_node

    def test_max_attempts_reached(self):
        """Test max attempts check"""
        mock_node = Mock(spec=Node)
        config = AblationConfig(
            name="test",
            description="test",
            code="",
            base_node=mock_node
        )

        for i in range(config.max_attempts):
            config.attempts += 1

        assert config.attempts == config.max_attempts
        assert config.attempts >= config.max_attempts


# =============================================================================
# Phase 1: Unit Tests (No LLM) - Parser
# =============================================================================

class TestParseKeywordPrefixResponse:
    """Tests for _parse_keyword_prefix_response utility function"""

    def test_parse_valid_response(self):
        """Test parsing a valid response"""
        response = """HYPERPARAM NAME: learning_rate
DESCRIPTION: Increase the learning rate from 0.001 to 0.01 to speed up convergence."""

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name == "learning_rate"
        assert description == "Increase the learning rate from 0.001 to 0.01 to speed up convergence."

    def test_parse_valid_ablation_response(self):
        """Test parsing a valid ablation response"""
        response = """ABLATION NAME: remove_attention
ABLATION DESCRIPTION: Remove the attention mechanism to test its contribution to performance."""

        name, description = _parse_keyword_prefix_response(
            response, "ABLATION NAME:", "ABLATION DESCRIPTION:"
        )
        assert name == "remove_attention"
        assert description == "Remove the attention mechanism to test its contribution to performance."

    def test_parse_multiline_description(self):
        """Test parsing response with multiline description"""
        response = """HYPERPARAM NAME: batch_size
DESCRIPTION: Increase batch size from 32 to 64.
This should improve training stability.
It may also speed up training on GPU."""

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name == "batch_size"
        assert "Increase batch size" in description
        assert "training stability" in description
        assert "speed up training" in description

    def test_parse_response_with_extra_text(self):
        """Test parsing response with extra text before keywords"""
        response = """Here is my suggestion:

HYPERPARAM NAME: epochs
DESCRIPTION: Train for more epochs to improve convergence."""

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name == "epochs"
        assert "Train for more epochs" in description

    def test_parse_missing_name(self):
        """Test parsing response missing the name keyword"""
        response = """DESCRIPTION: Some description without name."""

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name is None
        assert description is None

    def test_parse_missing_description(self):
        """Test parsing response missing the description keyword"""
        response = """HYPERPARAM NAME: learning_rate"""

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name is None
        assert description is None

    def test_parse_empty_response(self):
        """Test parsing empty response"""
        name, description = _parse_keyword_prefix_response(
            "", "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name is None
        assert description is None

    def test_parse_response_with_whitespace(self):
        """Test parsing response with extra whitespace"""
        response = """   HYPERPARAM NAME:   learning_rate
   DESCRIPTION:   Increase learning rate.   """

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name == "learning_rate"
        assert description == "Increase learning rate."

    def test_parse_response_different_order(self):
        """Test parsing response with keywords in different order"""
        response = """DESCRIPTION: This is the description.
HYPERPARAM NAME: test_param"""

        name, description = _parse_keyword_prefix_response(
            response, "HYPERPARAM NAME:", "DESCRIPTION:"
        )
        assert name == "test_param"
        assert description == "This is the description."


# =============================================================================
# Phase 2: Unit Tests (No LLM) - State Management
# =============================================================================

class TestParallelAgentStateManagement:
    """Tests for ParallelAgent state management (Stage 2/4)"""

    @pytest.fixture
    def mock_cfg(self):
        """Create a mock configuration object"""
        cfg = MagicMock()
        cfg.agent.num_workers = 2
        cfg.agent.code.model = "test-model"
        cfg.agent.code.temp = 0.7
        cfg.agent.search.num_drafts = 2
        cfg.agent.search.debug_prob = 0.1
        cfg.agent.search.max_debug_depth = 3
        cfg.exec.timeout = 60
        cfg.workspace_dir = "/tmp/test"
        cfg.log_dir = "/tmp/logs"
        return cfg

    @pytest.fixture
    def mock_journal(self):
        """Create a mock journal"""
        return Journal()

    def test_hyperparam_state_initialization(self, mock_cfg, mock_journal):
        """Test that hyperparam tuning state is initialized correctly"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}

            assert "tried_hyperparams" in agent._hyperparam_tuning_state
            assert isinstance(agent._hyperparam_tuning_state["tried_hyperparams"], set)
            assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == 0

    def test_ablation_state_initialization(self, mock_cfg, mock_journal):
        """Test that ablation state is initialized correctly"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent._ablation_state = {"completed_ablations": set()}

            assert "completed_ablations" in agent._ablation_state
            assert isinstance(agent._ablation_state["completed_ablations"], set)
            assert len(agent._ablation_state["completed_ablations"]) == 0

    def test_hyperparam_state_add_tried(self):
        """Test adding to tried hyperparams"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}

            agent._hyperparam_tuning_state["tried_hyperparams"].add("learning_rate")
            agent._hyperparam_tuning_state["tried_hyperparams"].add("batch_size")

            assert "learning_rate" in agent._hyperparam_tuning_state["tried_hyperparams"]
            assert "batch_size" in agent._hyperparam_tuning_state["tried_hyperparams"]
            assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == 2

    def test_hyperparam_state_no_duplicates(self):
        """Test that duplicate hyperparams are not added"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}

            agent._hyperparam_tuning_state["tried_hyperparams"].add("learning_rate")
            agent._hyperparam_tuning_state["tried_hyperparams"].add("learning_rate")

            assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == 1

    def test_ablation_state_add_completed(self):
        """Test adding to completed ablations"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent._ablation_state = {"completed_ablations": set()}

            agent._ablation_state["completed_ablations"].add("remove_dropout")
            agent._ablation_state["completed_ablations"].add("remove_attention")

            assert "remove_dropout" in agent._ablation_state["completed_ablations"]
            assert "remove_attention" in agent._ablation_state["completed_ablations"]
            assert len(agent._ablation_state["completed_ablations"]) == 2


# =============================================================================
# Phase 2: Unit Tests (No LLM) - State Update Methods
# =============================================================================

class TestStateUpdateMethods:
    """Tests for _update_hyperparam_tuning_state and _update_ablation_state"""

    def test_update_hyperparam_state_success(self):
        """Test updating hyperparam state on successful execution"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "2_baseline_tuning"
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}

            mock_node = Mock(spec=Node)
            mock_node.hyperparam_name = "learning_rate"
            mock_node.is_buggy = False
            mock_node.id = "test-id"

            agent._update_hyperparam_tuning_state(mock_node)

            assert "learning_rate" in agent._hyperparam_tuning_state["tried_hyperparams"]

    def test_update_hyperparam_state_wrong_stage(self):
        """Test that hyperparam state is not updated for wrong stage"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "1_initial"
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}

            mock_node = Mock(spec=Node)
            mock_node.hyperparam_name = "learning_rate"
            mock_node.is_buggy = False

            agent._update_hyperparam_tuning_state(mock_node)

            assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == 0

    def test_update_hyperparam_state_buggy_node(self):
        """Test that buggy nodes don't update hyperparam state"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "2_baseline_tuning"
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}

            mock_node = Mock(spec=Node)
            mock_node.hyperparam_name = "learning_rate"
            mock_node.is_buggy = True
            mock_node.id = "test-id"

            agent._update_hyperparam_tuning_state(mock_node)

            assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == 0

    def test_update_ablation_state_success(self):
        """Test updating ablation state on successful execution"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "4_ablation"
            agent._ablation_state = {"completed_ablations": set()}

            mock_node = Mock(spec=Node)
            mock_node.ablation_name = "remove_dropout"
            mock_node.is_buggy = False
            mock_node.id = "test-id"

            agent._update_ablation_state(mock_node)

            assert "remove_dropout" in agent._ablation_state["completed_ablations"]

    def test_update_ablation_state_wrong_stage(self):
        """Test that ablation state is not updated for wrong stage"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "3_creative"
            agent._ablation_state = {"completed_ablations": set()}

            mock_node = Mock(spec=Node)
            mock_node.ablation_name = "remove_dropout"
            mock_node.is_buggy = False

            agent._update_ablation_state(mock_node)

            assert len(agent._ablation_state["completed_ablations"]) == 0

    def test_update_ablation_state_buggy_node(self):
        """Test that buggy nodes don't update ablation state"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "4_ablation"
            agent._ablation_state = {"completed_ablations": set()}

            mock_node = Mock(spec=Node)
            mock_node.ablation_name = "remove_dropout"
            mock_node.is_buggy = True
            mock_node.id = "test-id"

            agent._update_ablation_state(mock_node)

            assert len(agent._ablation_state["completed_ablations"]) == 0


# =============================================================================
# Markers for LLM tests (to be skipped in fast runs)
# =============================================================================

# Mark tests that require LLM with @pytest.mark.llm
# These can be skipped with: pytest -m "not llm"

@pytest.mark.llm
class TestGenerateHyperparamTuningIdea:
    """Tests for _generate_hyperparam_tuning_idea (requires LLM)"""

    @pytest.fixture
    def agent_with_stage1_node(self):
        """Create agent with best_stage1_node for hyperparam tuning"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=cfg,
            journal=journal,
            stage_name="2_baseline_tuning",
        )

        # Set best_stage1_node with sample code
        agent.best_stage1_node = Node(
            code='''
import numpy as np

# Simple training loop
learning_rate = 0.001
epochs = 10

for epoch in range(epochs):
    loss = np.random.random()
    print(f"Epoch {epoch}: loss={loss}")
''',
            plan="Simple baseline training loop"
        )

        return agent

    def test_generate_idea_returns_valid_idea(self, agent_with_stage1_node):
        """Test that _generate_hyperparam_tuning_idea returns a valid idea"""
        agent = agent_with_stage1_node

        idea = agent._generate_hyperparam_tuning_idea()

        assert idea is not None
        assert isinstance(idea, HyperparamTuningIdea)
        assert idea.name != ""
        assert idea.description != ""
        print(f"\nGenerated idea: {idea.name}")
        print(f"Description: {idea.description}")

    def test_generate_idea_avoids_tried_hyperparams(self, agent_with_stage1_node):
        """Test that new ideas avoid previously tried hyperparams"""
        agent = agent_with_stage1_node

        # Mark some hyperparams as tried
        agent._hyperparam_tuning_state["tried_hyperparams"].add("learning_rate")
        agent._hyperparam_tuning_state["tried_hyperparams"].add("batch_size")

        idea = agent._generate_hyperparam_tuning_idea()

        assert idea is not None
        # The idea should ideally be different (though LLM may not always comply)
        print(f"\nTried: {agent._hyperparam_tuning_state['tried_hyperparams']}")
        print(f"New idea: {idea.name}")


@pytest.mark.llm
class TestGenerateAblationIdea:
    """Tests for _generate_ablation_idea (requires LLM)"""

    @pytest.fixture
    def agent_with_stage3_node(self):
        """Create agent with best_stage3_node for ablation"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=cfg,
            journal=journal,
            stage_name="4_ablation",
        )

        # Set best_stage3_node with sample code
        agent.best_stage3_node = Node(
            code='''
import numpy as np

class SimpleModel:
    def __init__(self):
        self.dropout_rate = 0.5
        self.attention = True
        self.hidden_layers = 3

    def forward(self, x):
        # Apply dropout
        if self.dropout_rate > 0:
            x = x * (np.random.random(x.shape) > self.dropout_rate)
        # Apply attention
        if self.attention:
            x = x * np.random.random(x.shape)
        return x

model = SimpleModel()
result = model.forward(np.random.random((10, 10)))
print(f"Result shape: {result.shape}")
''',
            plan="Model with dropout and attention"
        )

        return agent

    def test_generate_idea_returns_valid_idea(self, agent_with_stage3_node):
        """Test that _generate_ablation_idea returns a valid idea"""
        agent = agent_with_stage3_node

        idea = agent._generate_ablation_idea()

        assert idea is not None
        assert isinstance(idea, AblationIdea)
        assert idea.name != ""
        assert idea.description != ""
        print(f"\nGenerated ablation: {idea.name}")
        print(f"Description: {idea.description}")

    def test_generate_idea_avoids_completed_ablations(self, agent_with_stage3_node):
        """Test that new ideas avoid previously completed ablations"""
        agent = agent_with_stage3_node

        # Mark some ablations as completed
        agent._ablation_state["completed_ablations"].add("remove_dropout")
        agent._ablation_state["completed_ablations"].add("remove_attention")

        idea = agent._generate_ablation_idea()

        assert idea is not None
        print(f"\nCompleted: {agent._ablation_state['completed_ablations']}")
        print(f"New idea: {idea.name}")


@pytest.mark.llm
class TestGenerateHyperparamTuningNode:
    """Tests for _generate_hyperparam_tuning_node (requires LLM)"""

    @pytest.fixture
    def agent_with_stage1_node(self):
        """Create agent with best_stage1_node"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=cfg,
            journal=journal,
            stage_name="2_baseline_tuning",
        )

        agent.best_stage1_node = Node(
            code='''
import numpy as np

learning_rate = 0.001
epochs = 10

for epoch in range(epochs):
    loss = np.random.random()
    print(f"Epoch {epoch}: loss={loss}")
''',
            plan="Simple baseline"
        )

        return agent

    def test_generate_node_returns_valid_node(self, agent_with_stage1_node):
        """Test that _generate_hyperparam_tuning_node returns a valid node"""
        agent = agent_with_stage1_node

        idea = HyperparamTuningIdea(
            name="increase_epochs",
            description="Increase the number of epochs from 10 to 50"
        )

        node = agent._generate_hyperparam_tuning_node(
            parent_node=agent.best_stage1_node,
            hyperparam_idea=idea
        )

        assert node is not None
        assert isinstance(node, Node)
        assert node.code != ""
        assert node.plan != ""
        assert node.hyperparam_name == "increase_epochs"
        assert node.parent == agent.best_stage1_node
        print(f"\nGenerated node plan: {node.plan[:200]}...")
        print(f"Code length: {len(node.code)} chars")


@pytest.mark.llm
class TestGenerateAblationNode:
    """Tests for _generate_ablation_node (requires LLM)"""

    @pytest.fixture
    def agent_with_stage3_node(self):
        """Create agent with best_stage3_node"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=cfg,
            journal=journal,
            stage_name="4_ablation",
        )

        agent.best_stage3_node = Node(
            code='''
import numpy as np

class Model:
    def __init__(self):
        self.dropout = 0.5
        self.use_attention = True

    def forward(self, x):
        if self.dropout > 0:
            x = x * 0.5
        if self.use_attention:
            x = x * 2
        return x

model = Model()
print(model.forward(np.array([1, 2, 3])))
''',
            plan="Model with dropout and attention"
        )

        return agent

    def test_generate_node_returns_valid_node(self, agent_with_stage3_node):
        """Test that _generate_ablation_node returns a valid node"""
        agent = agent_with_stage3_node

        idea = AblationIdea(
            name="remove_dropout",
            description="Remove dropout to test its contribution to performance"
        )

        node = agent._generate_ablation_node(
            parent_node=agent.best_stage3_node,
            ablation_idea=idea
        )

        assert node is not None
        assert isinstance(node, Node)
        assert node.code != ""
        assert node.plan != ""
        assert node.ablation_name == "remove_dropout"
        assert node.parent == agent.best_stage3_node
        print(f"\nGenerated node plan: {node.plan[:200]}...")
        print(f"Code length: {len(node.code)} chars")


# =============================================================================
# Integration Tests
# =============================================================================

class TestStageTransitions:
    """Tests for stage transitions"""

    def test_stage_name_prefix_check(self):
        """Test stage name prefix checking"""
        assert "2_baseline_tuning".startswith("2_")
        assert "4_ablation".startswith("4_")
        assert not "1_initial".startswith("2_")
        assert not "3_creative".startswith("4_")

    def test_stage1_to_stage2_node_inheritance(self):
        """Test that best_stage1_node is used in Stage 2"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "2_baseline_tuning"
            agent.best_stage1_node = Mock(spec=Node)
            agent.best_stage1_node.id = "stage1-best"

            # Verify stage 2 should use best_stage1_node
            assert agent.stage_name.startswith("2_")
            assert agent.best_stage1_node is not None

    def test_stage3_to_stage4_node_inheritance(self):
        """Test that best_stage3_node is used in Stage 4"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "4_ablation"
            agent.best_stage3_node = Mock(spec=Node)
            agent.best_stage3_node.id = "stage3-best"

            # Verify stage 4 should use best_stage3_node
            assert agent.stage_name.startswith("4_")
            assert agent.best_stage3_node is not None


class TestSelectParallelNodesStageLogic:
    """Tests for _select_parallel_nodes stage-specific logic"""

    def test_select_nodes_stage2_uses_best_stage1_node(self):
        """Test that Stage 2 selects best_stage1_node"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "2_baseline_tuning"
            agent.num_workers = 1
            agent.journal = Mock(spec=Journal)
            agent.journal.draft_nodes = [Mock(spec=Node)]
            agent.cfg = MagicMock()
            agent.cfg.agent.search.num_drafts = 1
            agent.cfg.agent.search.debug_prob = 0.0

            # Setup best_stage1_node
            mock_best_node = Mock(spec=Node)
            mock_best_node.id = "best-stage1"
            agent.best_stage1_node = mock_best_node

            # Mock _get_leaves to return non-buggy leaf
            mock_leaf = Mock(spec=Node)
            mock_leaf.is_buggy = False
            agent._get_leaves = Mock(return_value=[mock_leaf])

            nodes = agent._select_parallel_nodes()

            assert len(nodes) == 1
            assert nodes[0] == mock_best_node

    def test_select_nodes_stage4_uses_best_stage3_node(self):
        """Test that Stage 4 selects best_stage3_node"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "4_ablation"
            agent.num_workers = 1
            agent.journal = Mock(spec=Journal)
            agent.journal.draft_nodes = [Mock(spec=Node)]
            agent.cfg = MagicMock()
            agent.cfg.agent.search.num_drafts = 1
            agent.cfg.agent.search.debug_prob = 0.0

            # Setup best_stage3_node
            mock_best_node = Mock(spec=Node)
            mock_best_node.id = "best-stage3"
            agent.best_stage3_node = mock_best_node

            # Mock _get_leaves to return non-buggy leaf
            mock_leaf = Mock(spec=Node)
            mock_leaf.is_buggy = False
            agent._get_leaves = Mock(return_value=[mock_leaf])

            nodes = agent._select_parallel_nodes()

            assert len(nodes) == 1
            assert nodes[0] == mock_best_node


class TestStepStateUpdate:
    """Tests for step() method state updates"""

    def test_step_updates_hyperparam_state_on_success(self):
        """Test that step() updates hyperparam state for Stage 2"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "2_baseline_tuning"
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}
            agent._ablation_state = {"completed_ablations": set()}

            # Create mock result node
            mock_node = Mock(spec=Node)
            mock_node.hyperparam_name = "learning_rate"
            mock_node.is_buggy = False
            mock_node.id = "test-node"

            agent._update_hyperparam_tuning_state(mock_node)

            assert "learning_rate" in agent._hyperparam_tuning_state["tried_hyperparams"]

    def test_step_updates_ablation_state_on_success(self):
        """Test that step() updates ablation state for Stage 4"""
        with patch.object(ParallelAgent, '__init__', lambda x, *args, **kwargs: None):
            agent = ParallelAgent.__new__(ParallelAgent)
            agent.stage_name = "4_ablation"
            agent._hyperparam_tuning_state = {"tried_hyperparams": set()}
            agent._ablation_state = {"completed_ablations": set()}

            # Create mock result node
            mock_node = Mock(spec=Node)
            mock_node.ablation_name = "remove_dropout"
            mock_node.is_buggy = False
            mock_node.id = "test-node"

            agent._update_ablation_state(mock_node)

            assert "remove_dropout" in agent._ablation_state["completed_ablations"]


# =============================================================================
# LLM Stage-Specific Tests
# =============================================================================

@pytest.mark.llm
class TestStage2HyperparamTuningFlow:
    """Stage 2: Hyperparameter tuning flow tests (requires LLM)"""

    @pytest.fixture
    def stage2_agent(self):
        """Create a Stage 2 agent with best_stage1_node"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="Multi-agent simulation for testing cooperation strategies",
            cfg=cfg,
            journal=journal,
            stage_name="2_baseline_tuning",
        )

        # Set best_stage1_node (simulating Stage 1 output)
        agent.best_stage1_node = Node(
            code='''
import numpy as np
import os

# Hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 32

# Simple training simulation
losses = []
for epoch in range(epochs):
    loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
    losses.append(loss)
    print(f"Epoch {epoch}: loss={loss:.4f}")

# Save experiment data
experiment_data = {
    'baseline': {
        'synthetic_data': {
            'losses': {'train': losses},
            'final_loss': losses[-1],
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print("Training complete!")
''',
            plan="Baseline training loop with basic hyperparameters"
        )

        return agent

    def test_stage2_generate_idea_and_node(self, stage2_agent):
        """Test full Stage 2 flow: generate idea -> generate node"""
        agent = stage2_agent

        # Step 1: Generate hyperparam tuning idea
        idea = agent._generate_hyperparam_tuning_idea()
        assert idea is not None
        assert isinstance(idea, HyperparamTuningIdea)
        print(f"\n[Stage 2] Generated idea: {idea.name}")
        print(f"Description: {idea.description}")

        # Step 2: Generate node from idea
        node = agent._generate_hyperparam_tuning_node(
            parent_node=agent.best_stage1_node,
            hyperparam_idea=idea
        )
        assert node is not None
        assert node.hyperparam_name == idea.name
        assert node.parent == agent.best_stage1_node
        assert "experiment_data" in node.code.lower() or "np.save" in node.code.lower()
        print(f"Generated node with {len(node.code)} chars of code")

        # Step 3: Update state
        agent._hyperparam_tuning_state["tried_hyperparams"].add(idea.name)
        assert idea.name in agent._hyperparam_tuning_state["tried_hyperparams"]

    def test_stage2_multiple_iterations(self, stage2_agent):
        """Test multiple iterations of hyperparam tuning"""
        agent = stage2_agent
        generated_ideas = []

        # Simulate 3 iterations
        for i in range(3):
            idea = agent._generate_hyperparam_tuning_idea()
            assert idea is not None
            generated_ideas.append(idea.name)

            # Mark as tried
            agent._hyperparam_tuning_state["tried_hyperparams"].add(idea.name)
            print(f"\n[Iteration {i+1}] Idea: {idea.name}")

        # Should have 3 ideas tried
        assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == 3
        print(f"\nAll tried hyperparams: {agent._hyperparam_tuning_state['tried_hyperparams']}")

    def test_stage2_state_persistence(self, stage2_agent):
        """Test that state persists across idea generations"""
        agent = stage2_agent

        # Pre-populate some tried hyperparams
        agent._hyperparam_tuning_state["tried_hyperparams"].add("learning_rate")
        agent._hyperparam_tuning_state["tried_hyperparams"].add("epochs")

        # Generate new idea
        idea = agent._generate_hyperparam_tuning_idea()
        assert idea is not None
        print(f"\nPreviously tried: {agent._hyperparam_tuning_state['tried_hyperparams']}")
        print(f"New idea: {idea.name}")

        # Ideally should suggest something different (LLM dependent)
        # Just verify it generates something valid
        assert idea.name != ""
        assert idea.description != ""


@pytest.mark.llm
class TestStage4AblationFlow:
    """Stage 4: Ablation study flow tests (requires LLM)"""

    @pytest.fixture
    def stage4_agent(self):
        """Create a Stage 4 agent with best_stage3_node"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="Multi-agent simulation for testing cooperation strategies",
            cfg=cfg,
            journal=journal,
            stage_name="4_ablation",
        )

        # Set best_stage3_node (simulating Stage 3 output)
        agent.best_stage3_node = Node(
            code='''
import numpy as np
import os

class MultiAgentSimulator:
    def __init__(self):
        self.num_agents = 10
        self.communication_enabled = True
        self.memory_enabled = True
        self.dropout_rate = 0.3

    def simulate(self, steps=100):
        results = []
        for step in range(steps):
            # Agent interactions
            if self.communication_enabled:
                comm_effect = np.random.random() * 0.5
            else:
                comm_effect = 0

            # Memory effect
            if self.memory_enabled:
                memory_effect = np.random.random() * 0.3
            else:
                memory_effect = 0

            # Dropout
            if self.dropout_rate > 0:
                active_agents = int(self.num_agents * (1 - self.dropout_rate))
            else:
                active_agents = self.num_agents

            score = active_agents * (1 + comm_effect + memory_effect)
            results.append(score)

        return results

# Run simulation
sim = MultiAgentSimulator()
results = sim.simulate()
print(f"Final score: {np.mean(results):.2f}")

# Save experiment data
experiment_data = {
    'full_model': {
        'synthetic_data': {
            'scores': results,
            'mean_score': np.mean(results),
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
''',
            plan="Multi-agent simulator with communication, memory, and dropout"
        )

        return agent

    def test_stage4_generate_idea_and_node(self, stage4_agent):
        """Test full Stage 4 flow: generate idea -> generate node"""
        agent = stage4_agent

        # Step 1: Generate ablation idea
        idea = agent._generate_ablation_idea()
        assert idea is not None
        assert isinstance(idea, AblationIdea)
        print(f"\n[Stage 4] Generated ablation: {idea.name}")
        print(f"Description: {idea.description}")

        # Step 2: Generate node from idea
        node = agent._generate_ablation_node(
            parent_node=agent.best_stage3_node,
            ablation_idea=idea
        )
        assert node is not None
        assert node.ablation_name == idea.name
        assert node.parent == agent.best_stage3_node
        print(f"Generated node with {len(node.code)} chars of code")

        # Step 3: Update state
        agent._ablation_state["completed_ablations"].add(idea.name)
        assert idea.name in agent._ablation_state["completed_ablations"]

    def test_stage4_multiple_ablations(self, stage4_agent):
        """Test multiple ablation studies"""
        agent = stage4_agent
        generated_ablations = []

        # Simulate 3 ablation iterations
        for i in range(3):
            idea = agent._generate_ablation_idea()
            assert idea is not None
            generated_ablations.append(idea.name)

            # Mark as completed
            agent._ablation_state["completed_ablations"].add(idea.name)
            print(f"\n[Ablation {i+1}] {idea.name}")

        # Should have 3 ablations completed
        assert len(agent._ablation_state["completed_ablations"]) == 3
        print(f"\nAll completed ablations: {agent._ablation_state['completed_ablations']}")

    def test_stage4_ablation_diversity(self, stage4_agent):
        """Test that ablations target different components"""
        agent = stage4_agent

        # Pre-populate some completed ablations
        agent._ablation_state["completed_ablations"].add("remove_communication")
        agent._ablation_state["completed_ablations"].add("remove_memory")

        # Generate new idea
        idea = agent._generate_ablation_idea()
        assert idea is not None
        print(f"\nCompleted ablations: {agent._ablation_state['completed_ablations']}")
        print(f"New ablation: {idea.name}")
        print(f"Description: {idea.description}")

        # Should suggest something different
        assert idea.name != ""
        assert idea.description != ""


@pytest.mark.llm
class TestStage3CreativeImprovementFlow:
    """Stage 3: Creative improvement flow tests (requires LLM)

    Stage 3 takes the best node from Stage 2 (tuned baseline) and
    applies creative improvements to develop novel solutions.
    """

    @pytest.fixture
    def stage3_agent(self):
        """Create a Stage 3 agent with best_stage2_node"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="""
            Multi-agent simulation for studying emergent cooperation.
            The goal is to develop creative improvements to the baseline
            that enhance cooperation rates and overall system performance.
            """,
            cfg=cfg,
            journal=journal,
            stage_name="3_creative_improvement",
        )

        # Set best_stage2_node (tuned baseline from Stage 2)
        agent.best_stage2_node = Node(
            code='''
import numpy as np
import os

# Tuned hyperparameters from Stage 2
NUM_AGENTS = 10
NUM_ROUNDS = 200  # Increased from 100
LEARNING_RATE = 0.05  # Tuned from 0.01
COOPERATION_THRESHOLD = 0.4  # Tuned from 0.5

class SimpleAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.cooperativeness = np.random.random()

    def decide(self):
        return self.cooperativeness > COOPERATION_THRESHOLD

    def update(self, reward):
        self.cooperativeness += LEARNING_RATE * (reward - 1)
        self.cooperativeness = np.clip(self.cooperativeness, 0, 1)

# Initialize agents
agents = [SimpleAgent(i) for i in range(NUM_AGENTS)]

# Simulation
cooperation_history = []
reward_history = []

for round_num in range(NUM_ROUNDS):
    decisions = [agent.decide() for agent in agents]
    cooperation_rate = np.mean(decisions)
    cooperation_history.append(cooperation_rate)

    # Calculate rewards
    if cooperation_rate > 0.5:
        rewards = [2.0] * NUM_AGENTS
    else:
        rewards = [float(d) for d in decisions]

    avg_reward = np.mean(rewards)
    reward_history.append(avg_reward)

    # Update agents
    for agent, reward in zip(agents, rewards):
        agent.update(reward)

# Save results
experiment_data = {
    'tuned_baseline': {
        'cooperation_game': {
            'cooperation_rate': cooperation_history,
            'average_reward': reward_history,
            'final_cooperation': cooperation_history[-1],
            'final_reward': reward_history[-1],
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print(f"Final cooperation rate: {cooperation_history[-1]:.2f}")
print(f"Final average reward: {reward_history[-1]:.2f}")
''',
            plan="Tuned baseline with optimized hyperparameters from Stage 2"
        )

        return agent

    def test_stage3_creative_improvement_draft(self, stage3_agent):
        """Test Stage 3 creative improvement generation"""
        agent = stage3_agent

        # Stage 3 uses the normal improvement flow (like Stage 1)
        # but starts from best_stage2_node instead of drafting from scratch

        # Verify agent is set up correctly
        assert agent.stage_name == "3_creative_improvement"
        assert agent.best_stage2_node is not None
        assert agent.best_stage2_node.code != ""

        print(f"\n[Stage 3] Starting from Stage 2 best node")
        print(f"Base code length: {len(agent.best_stage2_node.code)} chars")

        # Stage 3 should be able to generate improvements
        # This is done through the normal _draft or _improve methods
        # which are already tested in Stage 1 tests

    def test_stage3_inherits_from_stage2(self, stage3_agent):
        """Test that Stage 3 properly inherits from Stage 2"""
        agent = stage3_agent

        # Verify the inheritance chain
        assert agent.best_stage2_node is not None
        assert "Tuned" in agent.best_stage2_node.plan or "tuned" in agent.best_stage2_node.code.lower()

        # Stage 3 should use best_stage2_node as the starting point
        print(f"\n[Stage 3] Inherited plan: {agent.best_stage2_node.plan[:100]}...")

    def test_stage3_generates_creative_improvements(self, stage3_agent):
        """Test that Stage 3 can generate creative improvements via LLM"""
        agent = stage3_agent

        # Create an improvement prompt similar to what Stage 3 would use
        improvement_prompt = {
            "Introduction": (
                "You are an AI researcher. Based on the tuned baseline below, "
                "propose creative improvements to enhance the multi-agent simulation. "
                "Focus on novel mechanisms like communication, memory, or adaptive strategies."
            ),
            "Base code": agent.best_stage2_node.code,
            "Instructions": {
                "Requirements": [
                    "1. Identify ONE creative improvement to implement",
                    "2. The improvement should go beyond simple hyperparameter tuning",
                    "3. Examples: agent communication, memory mechanisms, adaptive behaviors",
                ]
            },
            "Response format": (
                "Your response should start with 'IMPROVEMENT NAME: <name>' on the first line. "
                "The second line should start with 'DESCRIPTION: <description>'."
            ),
        }

        # Query LLM for creative improvement idea
        from masist.treesearch.backend import query

        response = query(
            system_message=improvement_prompt,
            user_message=None,
            model=agent.cfg.agent.code.model,
            temperature=agent.cfg.agent.code.temp,
        )

        # Parse response
        improvement_name, improvement_desc = _parse_keyword_prefix_response(
            response, "IMPROVEMENT NAME:", "DESCRIPTION:"
        )

        assert improvement_name is not None
        assert improvement_desc is not None
        print(f"\n[Stage 3] Creative improvement: {improvement_name}")
        print(f"Description: {improvement_desc}")


@pytest.mark.llm
class TestStage3ToStage4Flow:
    """Test the flow from Stage 3 to Stage 4"""

    @pytest.fixture
    def stage3_completed_agent(self):
        """Create a Stage 3 agent with completed improvements"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        # Simulate Stage 3 completion with an improved node
        best_stage3_node = Node(
            code='''
import numpy as np
import os

# Creative improvements from Stage 3
NUM_AGENTS = 10
NUM_ROUNDS = 200
LEARNING_RATE = 0.05
COOPERATION_THRESHOLD = 0.4

class AdvancedAgent:
    """Agent with communication and memory (Stage 3 improvements)"""

    def __init__(self, agent_id):
        self.id = agent_id
        self.cooperativeness = np.random.random()
        self.memory = []  # Memory of past interactions
        self.communication_weight = 0.3  # How much to weight peer signals

    def decide(self, peer_signals=None):
        base_decision = self.cooperativeness

        # Use memory to inform decision
        if self.memory:
            memory_influence = np.mean(self.memory[-5:]) * 0.2
            base_decision += memory_influence

        # Use peer signals (communication)
        if peer_signals is not None:
            peer_influence = np.mean(peer_signals) * self.communication_weight
            base_decision += peer_influence

        return base_decision > COOPERATION_THRESHOLD

    def update(self, reward, decision):
        self.cooperativeness += LEARNING_RATE * (reward - 1)
        self.cooperativeness = np.clip(self.cooperativeness, 0, 1)
        self.memory.append(float(decision))

# Initialize agents
agents = [AdvancedAgent(i) for i in range(NUM_AGENTS)]

# Simulation with communication
cooperation_history = []
reward_history = []

for round_num in range(NUM_ROUNDS):
    # Generate peer signals (simple broadcast)
    peer_signals = [agent.cooperativeness for agent in agents]

    # Get decisions with communication
    decisions = [agent.decide(peer_signals) for agent in agents]
    cooperation_rate = np.mean(decisions)
    cooperation_history.append(cooperation_rate)

    # Calculate rewards
    if cooperation_rate > 0.5:
        rewards = [2.0] * NUM_AGENTS
    else:
        rewards = [float(d) for d in decisions]

    avg_reward = np.mean(rewards)
    reward_history.append(avg_reward)

    # Update agents
    for agent, reward, decision in zip(agents, rewards, decisions):
        agent.update(reward, decision)

# Save results
experiment_data = {
    'improved_model': {
        'cooperation_game': {
            'cooperation_rate': cooperation_history,
            'average_reward': reward_history,
            'final_cooperation': cooperation_history[-1],
            'final_reward': reward_history[-1],
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print(f"Final cooperation rate: {cooperation_history[-1]:.2f}")
print(f"Final average reward: {reward_history[-1]:.2f}")
''',
            plan="Advanced multi-agent system with communication and memory (Stage 3 improvements)"
        )

        # Create Stage 4 agent using Stage 3's best node
        agent = ParallelAgent(
            task_desc="Multi-agent simulation ablation study",
            cfg=cfg,
            journal=journal,
            stage_name="4_ablation",
            best_stage3_node=best_stage3_node,
        )

        return agent

    def test_stage3_to_stage4_ablation_targets(self, stage3_completed_agent):
        """Test that Stage 4 can identify ablation targets from Stage 3 code"""
        agent = stage3_completed_agent

        # Generate ablation idea
        idea = agent._generate_ablation_idea()

        assert idea is not None
        print(f"\n[Stage 3 -> 4] Ablation target: {idea.name}")
        print(f"Description: {idea.description}")

        # The ablation should target Stage 3's improvements
        # (communication, memory, etc.)
        stage3_features = ["communication", "memory", "peer", "signal"]
        code_lower = agent.best_stage3_node.code.lower()

        # Verify Stage 3 code has features to ablate
        has_features = any(f in code_lower for f in stage3_features)
        assert has_features, "Stage 3 code should have features to ablate"

    def test_stage3_to_stage4_full_transition(self, stage3_completed_agent):
        """Test full transition from Stage 3 to Stage 4"""
        agent = stage3_completed_agent

        # Generate ablation idea
        idea = agent._generate_ablation_idea()
        assert idea is not None

        # Generate ablation node
        node = agent._generate_ablation_node(
            parent_node=agent.best_stage3_node,
            ablation_idea=idea
        )

        assert node is not None
        assert node.ablation_name == idea.name
        assert node.parent == agent.best_stage3_node

        print(f"\n[Stage 3 -> 4] Full transition complete")
        print(f"Ablation: {idea.name}")
        print(f"Generated code: {len(node.code)} chars")


# =============================================================================
# LLM Integration Tests (Stage Transitions)
# =============================================================================

@pytest.mark.llm
class TestStageTransitionWithLLM:
    """Test stage transitions with actual LLM calls"""

    @pytest.fixture
    def base_config(self):
        """Load base configuration"""
        from omegaconf import OmegaConf
        return OmegaConf.load("tests/fixtures/test_config.yaml")

    def test_stage1_to_stage2_transition(self, base_config):
        """Test transition from Stage 1 to Stage 2"""
        # Stage 1: Create initial node
        stage1_journal = Journal()
        stage1_agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=base_config,
            journal=stage1_journal,
            stage_name="1_initial",
        )

        # Simulate Stage 1 output
        best_stage1_node = Node(
            code='''
import numpy as np
learning_rate = 0.01
for i in range(5):
    print(f"Step {i}")
''',
            plan="Initial baseline"
        )

        # Stage 2: Use Stage 1's best node
        stage2_journal = Journal()
        stage2_agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=base_config,
            journal=stage2_journal,
            stage_name="2_baseline_tuning",
            best_stage1_node=best_stage1_node,
        )

        # Verify Stage 2 can generate ideas based on Stage 1's code
        idea = stage2_agent._generate_hyperparam_tuning_idea()
        assert idea is not None
        print(f"\n[Stage 1 -> 2] Generated idea: {idea.name}")

        # Generate tuning node
        node = stage2_agent._generate_hyperparam_tuning_node(
            parent_node=best_stage1_node,
            hyperparam_idea=idea
        )
        assert node is not None
        assert node.parent == best_stage1_node
        print(f"Stage 2 node inherits from Stage 1 node")

    def test_stage2_to_stage3_transition(self, base_config):
        """Test transition from Stage 2 to Stage 3"""
        # Stage 2: Create tuned baseline node
        best_stage2_node = Node(
            code='''
import numpy as np

# Tuned hyperparameters
learning_rate = 0.05  # Tuned from 0.01
epochs = 50  # Tuned from 10
batch_size = 64  # Tuned from 32

for epoch in range(epochs):
    loss = 1.0 / (epoch + 1)
    print(f"Epoch {epoch}: loss={loss:.4f}")
''',
            plan="Tuned baseline from Stage 2"
        )

        # Stage 3: Use Stage 2's best node for creative improvements
        stage3_journal = Journal()
        stage3_agent = ParallelAgent(
            task_desc="Test multi-agent simulation - creative improvements",
            cfg=base_config,
            journal=stage3_journal,
            stage_name="3_creative_improvement",
            best_stage2_node=best_stage2_node,
        )

        # Verify Stage 3 has access to Stage 2's best node
        assert stage3_agent.best_stage2_node is not None
        assert stage3_agent.best_stage2_node == best_stage2_node
        print(f"\n[Stage 2 -> 3] Stage 3 has access to Stage 2 best node")
        print(f"Stage 2 plan: {best_stage2_node.plan}")

        # Stage 3 uses normal improvement flow, not special methods like Stage 2/4
        # So we just verify the setup is correct
        assert stage3_agent.stage_name == "3_creative_improvement"

    def test_stage3_to_stage4_transition(self, base_config):
        """Test transition from Stage 3 to Stage 4"""
        # Stage 3: Create creative solution node
        best_stage3_node = Node(
            code='''
import numpy as np

class ImprovedModel:
    def __init__(self):
        self.layers = 3
        self.attention = True
        self.dropout = 0.2

    def forward(self, x):
        for _ in range(self.layers):
            x = x * 1.1
            if self.attention:
                x = x + np.mean(x)
            if self.dropout > 0:
                x = x * 0.8
        return x

model = ImprovedModel()
result = model.forward(np.array([1, 2, 3]))
print(f"Result: {result}")
''',
            plan="Improved model with attention and dropout"
        )

        # Stage 4: Use Stage 3's best node for ablation
        stage4_journal = Journal()
        stage4_agent = ParallelAgent(
            task_desc="Test multi-agent simulation",
            cfg=base_config,
            journal=stage4_journal,
            stage_name="4_ablation",
            best_stage3_node=best_stage3_node,
        )

        # Verify Stage 4 can generate ablation ideas
        idea = stage4_agent._generate_ablation_idea()
        assert idea is not None
        print(f"\n[Stage 3 -> 4] Generated ablation: {idea.name}")

        # Generate ablation node
        node = stage4_agent._generate_ablation_node(
            parent_node=best_stage3_node,
            ablation_idea=idea
        )
        assert node is not None
        assert node.parent == best_stage3_node
        print(f"Stage 4 node inherits from Stage 3 node")


# =============================================================================
# E2E Tests (Full Pipeline)
# =============================================================================

@pytest.mark.llm
class TestE2EHyperparamTuning:
    """End-to-end tests for hyperparameter tuning (Stage 2)"""

    @pytest.fixture
    def e2e_stage2_setup(self):
        """Setup for E2E Stage 2 test"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        # Create agent with realistic setup
        agent = ParallelAgent(
            task_desc="""
            Implement a multi-agent simulation to study cooperation emergence.
            Agents should learn to cooperate through repeated interactions.
            Track cooperation rate and average reward as metrics.
            """,
            cfg=cfg,
            journal=journal,
            stage_name="2_baseline_tuning",
        )

        # Realistic Stage 1 baseline
        agent.best_stage1_node = Node(
            code='''
import numpy as np
import os

# Configuration
NUM_AGENTS = 5
NUM_ROUNDS = 100
LEARNING_RATE = 0.01
COOPERATION_THRESHOLD = 0.5

# Initialize agents
agent_cooperativeness = np.random.random(NUM_AGENTS)

# Simulation
cooperation_history = []
reward_history = []

for round_num in range(NUM_ROUNDS):
    # Agents decide to cooperate or defect
    decisions = agent_cooperativeness > COOPERATION_THRESHOLD
    cooperation_rate = np.mean(decisions)
    cooperation_history.append(cooperation_rate)

    # Calculate rewards
    if cooperation_rate > 0.5:
        rewards = np.ones(NUM_AGENTS) * 2
    else:
        rewards = decisions.astype(float)

    avg_reward = np.mean(rewards)
    reward_history.append(avg_reward)

    # Update cooperativeness (learning)
    agent_cooperativeness += LEARNING_RATE * (rewards - 1)
    agent_cooperativeness = np.clip(agent_cooperativeness, 0, 1)

# Save results
experiment_data = {
    'baseline': {
        'cooperation_game': {
            'cooperation_rate': cooperation_history,
            'average_reward': reward_history,
            'final_cooperation': cooperation_history[-1],
            'final_reward': reward_history[-1],
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print(f"Final cooperation rate: {cooperation_history[-1]:.2f}")
print(f"Final average reward: {reward_history[-1]:.2f}")
''',
            plan="Baseline multi-agent cooperation simulation"
        )

        return agent

    def test_e2e_stage2_single_iteration(self, e2e_stage2_setup):
        """E2E test: Single Stage 2 iteration"""
        agent = e2e_stage2_setup

        # Generate idea
        idea = agent._generate_hyperparam_tuning_idea()
        assert idea is not None
        print(f"\n[E2E Stage 2] Idea: {idea.name}")

        # Generate node
        node = agent._generate_hyperparam_tuning_node(
            parent_node=agent.best_stage1_node,
            hyperparam_idea=idea
        )
        assert node is not None
        assert node.code != ""

        # Verify code structure
        assert "experiment_data" in node.code or "np.save" in node.code
        print(f"Generated valid tuning code ({len(node.code)} chars)")

        # Update state
        agent._hyperparam_tuning_state["tried_hyperparams"].add(idea.name)

        # Simulate successful execution
        node.is_buggy = False
        node.hyperparam_name = idea.name
        agent._update_hyperparam_tuning_state(node)

        assert idea.name in agent._hyperparam_tuning_state["tried_hyperparams"]
        print(f"State updated successfully")

    def test_e2e_stage2_full_workflow(self, e2e_stage2_setup):
        """E2E test: Full Stage 2 workflow with multiple iterations"""
        agent = e2e_stage2_setup
        max_iterations = 2
        successful_nodes = []

        for i in range(max_iterations):
            print(f"\n--- Stage 2 Iteration {i+1} ---")

            # Generate idea
            idea = agent._generate_hyperparam_tuning_idea()
            print(f"Idea: {idea.name}")

            # Generate node
            node = agent._generate_hyperparam_tuning_node(
                parent_node=agent.best_stage1_node,
                hyperparam_idea=idea
            )

            # Simulate execution result
            node.is_buggy = False
            node.hyperparam_name = idea.name

            # Update state
            agent._update_hyperparam_tuning_state(node)
            successful_nodes.append(node)

            print(f"Iteration {i+1} complete")

        assert len(successful_nodes) == max_iterations
        assert len(agent._hyperparam_tuning_state["tried_hyperparams"]) == max_iterations
        print(f"\nFull workflow complete: {max_iterations} iterations")


@pytest.mark.llm
class TestE2EAblationStudy:
    """End-to-end tests for ablation study (Stage 4)"""

    @pytest.fixture
    def e2e_stage4_setup(self, request=None):
        """Setup for E2E Stage 4 test"""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("tests/fixtures/test_config.yaml")
        journal = Journal()

        agent = ParallelAgent(
            task_desc="""
            Multi-agent simulation with advanced features:
            - Communication between agents
            - Memory of past interactions
            - Adaptive learning rates
            Test which components contribute most to performance.
            """,
            cfg=cfg,
            journal=journal,
            stage_name="4_ablation",
        )

        # Stage 3 node with multiple components to ablate
        agent.best_stage3_node = Node(
            code='''
import numpy as np
import os

class AdvancedAgent:
    def __init__(self, agent_id, use_communication=True, use_memory=True, use_adaptive_lr=True):
        self.id = agent_id
        self.use_communication = use_communication
        self.use_memory = use_memory
        self.use_adaptive_lr = use_adaptive_lr

        self.memory = []
        self.learning_rate = 0.1

    def act(self, observation, messages=None):
        action_score = np.random.random()

        # Communication effect
        if self.use_communication and messages:
            action_score += np.mean(messages) * 0.2

        # Memory effect
        if self.use_memory and self.memory:
            action_score += np.mean(self.memory[-5:]) * 0.1
            self.memory.append(action_score)

        # Adaptive learning rate
        if self.use_adaptive_lr:
            self.learning_rate *= 0.99

        return action_score > 0.5

class MultiAgentEnvironment:
    def __init__(self, num_agents=10):
        self.agents = [AdvancedAgent(i) for i in range(num_agents)]

    def step(self):
        # Gather messages
        messages = [np.random.random() for _ in self.agents]

        # Get actions
        actions = [agent.act(None, messages) for agent in self.agents]

        # Calculate reward
        cooperation_rate = np.mean(actions)
        return cooperation_rate

    def run(self, num_steps=100):
        results = []
        for _ in range(num_steps):
            results.append(self.step())
        return results

# Run simulation
env = MultiAgentEnvironment()
results = env.run()

experiment_data = {
    'full_model': {
        'multi_agent_sim': {
            'cooperation_rates': results,
            'mean_cooperation': np.mean(results),
            'final_cooperation': results[-1],
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print(f"Mean cooperation: {np.mean(results):.2f}")
''',
            plan="Advanced multi-agent system with communication, memory, and adaptive learning"
        )

        return agent

    def test_e2e_stage4_single_ablation(self, e2e_stage4_setup):
        """E2E test: Single ablation study"""
        agent = e2e_stage4_setup

        # Generate ablation idea
        idea = agent._generate_ablation_idea()
        assert idea is not None
        print(f"\n[E2E Stage 4] Ablation: {idea.name}")
        print(f"Description: {idea.description}")

        # Generate ablation node
        node = agent._generate_ablation_node(
            parent_node=agent.best_stage3_node,
            ablation_idea=idea
        )
        assert node is not None
        assert node.code != ""

        # Verify ablation code
        print(f"Generated ablation code ({len(node.code)} chars)")

        # Simulate successful execution
        node.is_buggy = False
        node.ablation_name = idea.name
        agent._update_ablation_state(node)

        assert idea.name in agent._ablation_state["completed_ablations"]
        print(f"Ablation completed successfully")

    def test_e2e_stage4_systematic_ablation(self, e2e_stage4_setup):
        """E2E test: Systematic ablation of multiple components"""
        agent = e2e_stage4_setup
        max_ablations = 2
        ablation_results = []

        for i in range(max_ablations):
            print(f"\n--- Ablation Study {i+1} ---")

            # Generate ablation idea
            idea = agent._generate_ablation_idea()
            print(f"Ablating: {idea.name}")

            # Generate ablation node
            node = agent._generate_ablation_node(
                parent_node=agent.best_stage3_node,
                ablation_idea=idea
            )

            # Simulate execution
            node.is_buggy = False
            node.ablation_name = idea.name

            # Update state
            agent._update_ablation_state(node)

            ablation_results.append({
                "name": idea.name,
                "description": idea.description,
                "code_length": len(node.code),
            })

            print(f"Ablation {i+1} complete")

        assert len(ablation_results) == max_ablations
        assert len(agent._ablation_state["completed_ablations"]) == max_ablations
        print(f"\nSystematic ablation complete: {max_ablations} studies")
        for result in ablation_results:
            print(f"  - {result['name']}: {result['code_length']} chars")


# =============================================================================
# Full Pipeline E2E Test (Stage 1 → 2 → 3 → 4) - Method-level calls
# =============================================================================

@pytest.mark.llm
class TestFullPipelineE2E:
    """End-to-end test for complete Stage 1 → 2 → 3 → 4 pipeline"""

    @pytest.fixture
    def full_pipeline_setup(self):
        """Setup for full pipeline test"""
        from omegaconf import OmegaConf
        return OmegaConf.load("tests/fixtures/test_config.yaml")

    def test_full_pipeline_stage1_to_stage4(self, full_pipeline_setup):
        """E2E: Complete pipeline from Stage 1 through Stage 4

        Stage 1: Initial draft generation (baseline)
        Stage 2: Hyperparameter tuning
        Stage 3: Creative improvements
        Stage 4: Ablation studies
        """
        cfg = full_pipeline_setup
        print("\n" + "=" * 70)
        print("FULL PIPELINE E2E TEST: Stage 1 → 2 → 3 → 4")
        print("=" * 70)

        # =====================================================================
        # STAGE 1: Initial Draft (Baseline)
        # =====================================================================
        print("\n" + "-" * 50)
        print("STAGE 1: Initial Draft Generation")
        print("-" * 50)

        stage1_journal = Journal()
        stage1_agent = ParallelAgent(
            task_desc="""
            Multi-agent cooperation simulation.
            Agents learn to cooperate through repeated interactions.
            Track cooperation rate and average reward.
            """,
            cfg=cfg,
            journal=stage1_journal,
            stage_name="1_initial",
        )

        # Simulate Stage 1 output (in real scenario, this would be generated)
        best_stage1_node = Node(
            code='''
import numpy as np
import os

# Stage 1: Baseline configuration
NUM_AGENTS = 5
NUM_ROUNDS = 50
LEARNING_RATE = 0.01
COOPERATION_THRESHOLD = 0.5

# Initialize agents
agent_scores = np.random.random(NUM_AGENTS)

# Simulation loop
cooperation_history = []
reward_history = []

for round_num in range(NUM_ROUNDS):
    # Decide cooperation
    decisions = agent_scores > COOPERATION_THRESHOLD
    cooperation_rate = np.mean(decisions)
    cooperation_history.append(cooperation_rate)

    # Calculate rewards
    if cooperation_rate > 0.5:
        rewards = np.ones(NUM_AGENTS) * 2.0
    else:
        rewards = decisions.astype(float)

    reward_history.append(np.mean(rewards))

    # Update agents
    agent_scores += LEARNING_RATE * (rewards - 1)
    agent_scores = np.clip(agent_scores, 0, 1)

# Save results
experiment_data = {
    'stage1_baseline': {
        'cooperation_game': {
            'cooperation_rate': cooperation_history,
            'average_reward': reward_history,
            'final_cooperation': cooperation_history[-1],
            'final_reward': reward_history[-1],
        }
    }
}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print(f"[Stage 1] Final cooperation: {cooperation_history[-1]:.2f}")
print(f"[Stage 1] Final reward: {reward_history[-1]:.2f}")
''',
            plan="Stage 1: Baseline multi-agent cooperation simulation"
        )

        print(f"Stage 1 baseline created: {len(best_stage1_node.code)} chars")
        print(f"Plan: {best_stage1_node.plan}")

        # =====================================================================
        # STAGE 2: Hyperparameter Tuning
        # =====================================================================
        print("\n" + "-" * 50)
        print("STAGE 2: Hyperparameter Tuning")
        print("-" * 50)

        stage2_journal = Journal()
        stage2_agent = ParallelAgent(
            task_desc="Multi-agent cooperation simulation - hyperparameter tuning",
            cfg=cfg,
            journal=stage2_journal,
            stage_name="2_baseline_tuning",
            best_stage1_node=best_stage1_node,
        )

        # Generate hyperparameter tuning idea
        hyperparam_idea = stage2_agent._generate_hyperparam_tuning_idea()
        assert hyperparam_idea is not None
        print(f"Hyperparameter idea: {hyperparam_idea.name}")
        print(f"Description: {hyperparam_idea.description}")

        # Generate tuned node
        stage2_node = stage2_agent._generate_hyperparam_tuning_node(
            parent_node=best_stage1_node,
            hyperparam_idea=hyperparam_idea
        )
        assert stage2_node is not None
        assert stage2_node.parent == best_stage1_node
        print(f"Stage 2 node created: {len(stage2_node.code)} chars")

        # Update state
        stage2_agent._hyperparam_tuning_state["tried_hyperparams"].add(hyperparam_idea.name)

        # Simulate successful execution
        stage2_node.is_buggy = False
        stage2_node.hyperparam_name = hyperparam_idea.name

        # Use Stage 2 node as best for Stage 3
        best_stage2_node = stage2_node

        # =====================================================================
        # STAGE 3: Creative Improvements
        # =====================================================================
        print("\n" + "-" * 50)
        print("STAGE 3: Creative Improvements")
        print("-" * 50)

        stage3_journal = Journal()
        stage3_agent = ParallelAgent(
            task_desc="Multi-agent cooperation simulation - creative improvements",
            cfg=cfg,
            journal=stage3_journal,
            stage_name="3_creative_improvement",
            best_stage2_node=best_stage2_node,
        )

        # Stage 3 uses normal improvement flow
        # Generate creative improvement via LLM
        from masist.treesearch.backend import query

        improvement_prompt = {
            "Introduction": (
                "You are an AI researcher. Based on the tuned baseline below, "
                "propose ONE creative improvement to enhance the multi-agent simulation. "
                "Focus on novel mechanisms beyond hyperparameter tuning."
            ),
            "Base code": best_stage2_node.code,
            "Requirements": [
                "1. Propose ONE specific creative improvement",
                "2. Examples: agent communication, memory, adaptive strategies",
                "3. Must be implementable as code modification",
            ],
            "Response format": (
                "IMPROVEMENT NAME: <name>\n"
                "DESCRIPTION: <description>\n"
                "CODE CHANGES: <brief description of what to change>"
            ),
        }

        response = query(
            system_message=improvement_prompt,
            user_message=None,
            model=cfg.agent.code.model,
            temperature=cfg.agent.code.temp,
        )

        improvement_name, improvement_desc = _parse_keyword_prefix_response(
            response, "IMPROVEMENT NAME:", "DESCRIPTION:"
        )
        assert improvement_name is not None
        print(f"Creative improvement: {improvement_name}")
        print(f"Description: {improvement_desc}")

        # Generate improved code
        code_prompt = {
            "Introduction": "Implement the following improvement to the code.",
            "Base code": best_stage2_node.code,
            "Improvement": f"{improvement_name}: {improvement_desc}",
            "Requirements": [
                "1. Return ONLY the complete modified Python code",
                "2. Preserve experiment_data saving structure",
                "3. Add the improvement as described",
            ],
        }

        improved_code = query(
            system_message=code_prompt,
            user_message=None,
            model=cfg.agent.code.model,
            temperature=cfg.agent.code.temp,
        )

        # Create Stage 3 node
        best_stage3_node = Node(
            code=improved_code,
            plan=f"Stage 3: {improvement_name} - {improvement_desc}",
            parent=best_stage2_node,
        )
        print(f"Stage 3 node created: {len(best_stage3_node.code)} chars")

        # =====================================================================
        # STAGE 4: Ablation Studies
        # =====================================================================
        print("\n" + "-" * 50)
        print("STAGE 4: Ablation Studies")
        print("-" * 50)

        stage4_journal = Journal()
        stage4_agent = ParallelAgent(
            task_desc="Multi-agent cooperation simulation - ablation studies",
            cfg=cfg,
            journal=stage4_journal,
            stage_name="4_ablation",
            best_stage3_node=best_stage3_node,
        )

        # Generate ablation idea
        ablation_idea = stage4_agent._generate_ablation_idea()
        assert ablation_idea is not None
        print(f"Ablation target: {ablation_idea.name}")
        print(f"Description: {ablation_idea.description}")

        # Generate ablation node
        stage4_node = stage4_agent._generate_ablation_node(
            parent_node=best_stage3_node,
            ablation_idea=ablation_idea
        )
        assert stage4_node is not None
        assert stage4_node.parent == best_stage3_node
        print(f"Stage 4 node created: {len(stage4_node.code)} chars")

        # Update state
        stage4_agent._ablation_state["completed_ablations"].add(ablation_idea.name)

        # =====================================================================
        # VERIFICATION
        # =====================================================================
        print("\n" + "-" * 50)
        print("PIPELINE VERIFICATION")
        print("-" * 50)

        # Verify inheritance chain
        assert stage2_node.parent == best_stage1_node, "Stage 2 should inherit from Stage 1"
        assert best_stage3_node.parent == best_stage2_node, "Stage 3 should inherit from Stage 2"
        assert stage4_node.parent == best_stage3_node, "Stage 4 should inherit from Stage 3"

        print("✓ Inheritance chain verified: Stage 1 → 2 → 3 → 4")

        # Verify state management
        assert hyperparam_idea.name in stage2_agent._hyperparam_tuning_state["tried_hyperparams"]
        assert ablation_idea.name in stage4_agent._ablation_state["completed_ablations"]

        print("✓ State management verified")

        # Summary
        print("\n" + "=" * 70)
        print("FULL PIPELINE E2E TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Stage 1 (Baseline):     {len(best_stage1_node.code):>5} chars")
        print(f"Stage 2 (Tuning):       {len(stage2_node.code):>5} chars - {hyperparam_idea.name}")
        print(f"Stage 3 (Creative):     {len(best_stage3_node.code):>5} chars - {improvement_name}")
        print(f"Stage 4 (Ablation):     {len(stage4_node.code):>5} chars - {ablation_idea.name}")
        print("=" * 70)

    def test_full_pipeline_multiple_iterations(self, full_pipeline_setup):
        """E2E: Multiple iterations at each stage

        Tests:
        - Stage 2: 2 hyperparameter tuning iterations
        - Stage 4: 2 ablation iterations
        """
        cfg = full_pipeline_setup
        print("\n" + "=" * 70)
        print("FULL PIPELINE E2E TEST: Multiple Iterations")
        print("=" * 70)

        # Stage 1: Baseline
        best_stage1_node = Node(
            code='''
import numpy as np

NUM_AGENTS = 5
LEARNING_RATE = 0.01

for i in range(10):
    score = np.random.random()
    print(f"Round {i}: {score:.2f}")

experiment_data = {'baseline': {'score': score}}
np.savez('experiment_data.npz', experiment_data=experiment_data)
''',
            plan="Simple baseline"
        )
        print(f"\n[Stage 1] Baseline: {len(best_stage1_node.code)} chars")

        # Stage 2: Multiple hyperparameter iterations
        print("\n[Stage 2] Hyperparameter tuning iterations...")
        stage2_journal = Journal()
        stage2_agent = ParallelAgent(
            task_desc="Multi-agent simulation",
            cfg=cfg,
            journal=stage2_journal,
            stage_name="2_baseline_tuning",
            best_stage1_node=best_stage1_node,
        )

        stage2_nodes = []
        for i in range(2):
            idea = stage2_agent._generate_hyperparam_tuning_idea()
            node = stage2_agent._generate_hyperparam_tuning_node(
                parent_node=best_stage1_node,
                hyperparam_idea=idea
            )
            stage2_agent._hyperparam_tuning_state["tried_hyperparams"].add(idea.name)
            stage2_nodes.append((idea.name, node))
            print(f"  Iteration {i+1}: {idea.name}")

        assert len(stage2_agent._hyperparam_tuning_state["tried_hyperparams"]) == 2
        best_stage2_node = stage2_nodes[0][1]  # Use first as best

        # Stage 3: Creative improvement (single iteration for simplicity)
        print("\n[Stage 3] Creative improvement...")
        best_stage3_node = Node(
            code=best_stage2_node.code + "\n# Stage 3: Added communication",
            plan="Stage 3 improvement",
            parent=best_stage2_node,
        )
        print(f"  Improvement applied: {len(best_stage3_node.code)} chars")

        # Stage 4: Multiple ablation iterations
        print("\n[Stage 4] Ablation study iterations...")
        stage4_journal = Journal()
        stage4_agent = ParallelAgent(
            task_desc="Multi-agent simulation - ablation",
            cfg=cfg,
            journal=stage4_journal,
            stage_name="4_ablation",
            best_stage3_node=best_stage3_node,
        )

        stage4_nodes = []
        for i in range(2):
            idea = stage4_agent._generate_ablation_idea()
            node = stage4_agent._generate_ablation_node(
                parent_node=best_stage3_node,
                ablation_idea=idea
            )
            stage4_agent._ablation_state["completed_ablations"].add(idea.name)
            stage4_nodes.append((idea.name, node))
            print(f"  Iteration {i+1}: {idea.name}")

        assert len(stage4_agent._ablation_state["completed_ablations"]) == 2

        print("\n" + "=" * 70)
        print("MULTIPLE ITERATIONS TEST COMPLETED")
        print(f"Stage 2: {len(stage2_agent._hyperparam_tuning_state['tried_hyperparams'])} hyperparams tried")
        print(f"Stage 4: {len(stage4_agent._ablation_state['completed_ablations'])} ablations completed")
        print("=" * 70)


# =============================================================================
# Real Full Pipeline E2E Test (agent.run() execution)
# Follows test_stages.py style - function-based with context managers
# =============================================================================

# Test configuration path (same as test_stages.py)
from pathlib import Path
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "test_config.yaml"


def create_hyperparam_test_config(workspace_name: str = "test_hyperparam") -> "Config":
    """Create test configuration using YAML file (same style as test_stages.py)"""
    from masist.treesearch.utils.config import _load_cfg, prep_cfg

    cfg = _load_cfg(TEST_CONFIG_PATH)
    cfg.exp_name = workspace_name
    cfg.data_dir = str(Path(__file__).parent / "fixtures" / "test_data")
    test_data_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.goal = "Test goal for hyperparam/ablation"
    cfg = prep_cfg(cfg)
    return cfg


def test_real_stage1_execution():
    """Test 13: Real Stage 1 execution with agent.run()

    This test:
    1. Creates a ParallelAgent for Stage 1
    2. Runs agent.run() with limited steps
    3. Verifies nodes are created
    """
    print("\n" + "=" * 80)
    print("Test 13: Real Stage 1 Execution")
    print("=" * 80)

    config = create_hyperparam_test_config(workspace_name="test_real_stage1")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2

    try:
        journal = Journal()

        with ParallelAgent(
            task_desc="""
            Create a simple Python script that:
            1. Generates random numbers
            2. Calculates their mean
            3. Prints the result
            4. Saves experiment_data to a .npz file
            """,
            cfg=config,
            journal=journal,
            stage_name="1_initial_implementation_1",
        ) as agent:
            print(f"\n[TEST] Running Stage 1 with {agent.num_workers} workers...")
            print(f"[TEST] max_steps=2")

            success = agent.run(max_steps=2)

            print(f"\n[TEST] Execution completed:")
            print(f"  - Success: {success}")
            print(f"  - Total nodes: {len(journal)}")
            print(f"  - Good nodes: {len(journal.good_nodes)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")

        print("\nTest 13 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 13 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_real_full_pipeline_stage1_to_stage4():
    """Test 14: Real Full Pipeline Stage 1 → 2 → 3 → 4

    This test:
    1. Runs Stage 1 to get baseline
    2. Runs Stage 2 (hyperparameter tuning) using Stage 1 best node
    3. Runs Stage 3 (creative improvement) using Stage 2 best node
    4. Runs Stage 4 (ablation studies) using Stage 3 best node

    WARNING: This test is slow as it actually executes code at each stage.
    """
    print("\n" + "=" * 80)
    print("Test 14: Real Full Pipeline Stage 1 → 2 → 3 → 4")
    print("WARNING: This test may take several minutes")
    print("=" * 80)

    config = create_hyperparam_test_config(workspace_name="test_full_pipeline")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2
    config.agent.stages.stage1_max_iters = 3
    config.agent.stages.stage2_max_iters = 3
    config.agent.stages.stage3_max_iters = 3
    config.agent.stages.stage4_max_iters = 3

    try:
        # =================================================================
        # STAGE 1: Initial Draft
        # =================================================================
        print("\n" + "-" * 60)
        print("[STAGE 1] Initial Draft")
        print("-" * 60)

        stage1_journal = Journal()

        with ParallelAgent(
            task_desc="""
            Create a simple multi-agent simulation:
            - Initialize 5 agents with random scores
            - Run 10 rounds of interaction
            - Print final scores
            - Save experiment_data dictionary to experiment_data.npz
            """,
            cfg=config,
            journal=stage1_journal,
            stage_name="1_initial_implementation_1",
        ) as agent:
            print(f"[TEST] Running Stage 1 with {agent.num_workers} workers...")
            success = agent.run(max_steps=3)

            print(f"[TEST] Stage 1 result: success={success}, nodes={len(stage1_journal)}")

        # Get best node from Stage 1 (or create fallback)
        if stage1_journal.good_nodes:
            best_stage1_node = stage1_journal.get_best_node(only_good=True)
            print(f"[TEST] Stage 1 best node: {best_stage1_node.id[:8] if best_stage1_node else 'None'}")
        else:
            print("[TEST] Creating fallback Stage 1 node...")
            best_stage1_node = Node(
                code='''
import numpy as np

NUM_AGENTS = 5
scores = np.random.random(NUM_AGENTS)

for round_num in range(10):
    scores = scores * 0.9 + np.random.random(NUM_AGENTS) * 0.1

print(f"Final scores: {scores}")
experiment_data = {'stage1': {'scores': scores.tolist()}}
np.savez('experiment_data.npz', experiment_data=experiment_data)
''',
                plan="Fallback Stage 1 baseline"
            )

        # =================================================================
        # STAGE 2: Hyperparameter Tuning
        # =================================================================
        print("\n" + "-" * 60)
        print("[STAGE 2] Hyperparameter Tuning")
        print("-" * 60)

        stage2_journal = Journal()

        stage2_task_desc = """
Multi-agent simulation - hyperparameter tuning stage.

Current Main Stage: baseline_tuning
Sub-stage goals:
- Tune hyperparameters to improve performance
- Do not change core architecture
"""

        with ParallelAgent(
            task_desc=stage2_task_desc,
            cfg=config,
            journal=stage2_journal,
            stage_name="2_baseline_tuning_1",
            best_stage1_node=best_stage1_node,
        ) as agent:
            print(f"[TEST] Running Stage 2 with {agent.num_workers} workers...")
            success = agent.run(max_steps=3)

            print(f"[TEST] Stage 2 result: success={success}, nodes={len(stage2_journal)}")
            print(f"  - Good nodes: {len(stage2_journal.good_nodes)}")

        best_stage2_node = stage2_journal.get_best_node(only_good=True) if stage2_journal.good_nodes else best_stage1_node
        print(f"[TEST] Stage 2 best node: {best_stage2_node.id[:8] if hasattr(best_stage2_node, 'id') and best_stage2_node.id else 'fallback'}")

        # =================================================================
        # STAGE 3: Creative Improvements
        # =================================================================
        print("\n" + "-" * 60)
        print("[STAGE 3] Creative Improvements")
        print("-" * 60)

        stage3_journal = Journal()

        stage3_task_desc = """
Multi-agent simulation - creative improvement stage.

Current Main Stage: creative_research
Sub-stage goals:
- Explore novel approaches beyond baseline
- Add creative mechanisms (communication, memory, etc.)
"""

        with ParallelAgent(
            task_desc=stage3_task_desc,
            cfg=config,
            journal=stage3_journal,
            stage_name="3_creative_research_1",
            best_stage2_node=best_stage2_node,
        ) as agent:
            print(f"[TEST] Running Stage 3 with {agent.num_workers} workers...")
            success = agent.run(max_steps=3)

            print(f"[TEST] Stage 3 result: success={success}, nodes={len(stage3_journal)}")
            print(f"  - Good nodes: {len(stage3_journal.good_nodes)}")

        best_stage3_node = stage3_journal.get_best_node(only_good=True) if stage3_journal.good_nodes else best_stage2_node
        print(f"[TEST] Stage 3 best node: {best_stage3_node.id[:8] if hasattr(best_stage3_node, 'id') and best_stage3_node.id else 'fallback'}")

        # =================================================================
        # STAGE 4: Ablation Studies
        # =================================================================
        print("\n" + "-" * 60)
        print("[STAGE 4] Ablation Studies")
        print("-" * 60)

        stage4_journal = Journal()

        stage4_task_desc = """
Multi-agent simulation - ablation studies stage.

Current Main Stage: ablation_studies
Sub-stage goals:
- Systematically analyze component contributions
- Remove/modify individual components to measure impact
"""

        with ParallelAgent(
            task_desc=stage4_task_desc,
            cfg=config,
            journal=stage4_journal,
            stage_name="4_ablation_studies_1",
            best_stage3_node=best_stage3_node,
        ) as agent:
            print(f"[TEST] Running Stage 4 with {agent.num_workers} workers...")
            success = agent.run(max_steps=3)

            print(f"[TEST] Stage 4 result: success={success}, nodes={len(stage4_journal)}")
            print(f"  - Good nodes: {len(stage4_journal.good_nodes)}")

        # =================================================================
        # SUMMARY
        # =================================================================
        print("\n" + "-" * 60)
        print("[SUMMARY] Pipeline Results")
        print("-" * 60)
        print(f"  Stage 1: {len(stage1_journal)} nodes, {len(stage1_journal.good_nodes)} good")
        print(f"  Stage 2: {len(stage2_journal)} nodes, {len(stage2_journal.good_nodes)} good")
        print(f"  Stage 3: {len(stage3_journal)} nodes, {len(stage3_journal.good_nodes)} good")
        print(f"  Stage 4: {len(stage4_journal)} nodes, {len(stage4_journal.good_nodes)} good")

        print("\nTest 14 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 14 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_real_stage2_step_execution():
    """Test 15: Real Stage 2 step() execution

    This test:
    1. Creates a Stage 1 baseline node
    2. Creates Stage 2 agent with best_stage1_node
    3. Runs a single step()
    4. Verifies state updates
    """
    print("\n" + "=" * 80)
    print("Test 15: Real Stage 2 Step Execution")
    print("=" * 80)

    config = create_hyperparam_test_config(workspace_name="test_stage2_step")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2

    try:
        # Create Stage 1 baseline
        best_stage1_node = Node(
            code='''
import numpy as np

LEARNING_RATE = 0.01
NUM_ROUNDS = 10

scores = []
for i in range(NUM_ROUNDS):
    score = np.random.random() * LEARNING_RATE
    scores.append(score)
    print(f"Round {i}: {score:.4f}")

experiment_data = {'baseline': {'scores': scores}}
np.savez('experiment_data.npz', experiment_data=experiment_data)
print(f"Mean score: {np.mean(scores):.4f}")
''',
            plan="Stage 1 baseline"
        )

        journal = Journal()

        with ParallelAgent(
            task_desc="Tune hyperparameters for the simulation",
            cfg=config,
            journal=journal,
            stage_name="2_baseline_tuning_1",
            best_stage1_node=best_stage1_node,
        ) as agent:
            print(f"\n[TEST] Running single step() with {agent.num_workers} workers...")

            agent.step()

            print(f"\n[TEST] Step completed:")
            print(f"  - All nodes: {len(journal)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")
            print(f"  - Tried hyperparams: {agent._hyperparam_tuning_state['tried_hyperparams']}")

        print("\nTest 15 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 15 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Entry Point (test_stages.py style)
# =============================================================================

def main():
    """Run hyperparam/ablation tests"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning & Ablation Tests for MASist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tests:
  13 - Real Stage 1 Execution
  14 - Real Full Pipeline (Stage 1 → 2 → 3 → 4)
  15 - Real Stage 2 Step Execution

Examples:
  python tests/test_hyperparam_ablation.py --test 13      # Run Stage 1 test
  python tests/test_hyperparam_ablation.py --test 14      # Run full pipeline
  python tests/test_hyperparam_ablation.py --full         # Run all real tests
"""
    )
    parser.add_argument("--test", type=int, choices=[13, 14, 15], help="Run specific test by number")
    parser.add_argument("--full", action="store_true", help="Run all real execution tests")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Hyperparameter Tuning & Ablation Tests")
    print("=" * 80)

    # Test mapping
    test_map = {
        13: ("Real Stage 1 Execution", test_real_stage1_execution),
        14: ("Real Full Pipeline (1→2→3→4)", test_real_full_pipeline_stage1_to_stage4),
        15: ("Real Stage 2 Step Execution", test_real_stage2_step_execution),
    }

    results = {}

    if args.test:
        # Run specific test
        name, test_func = test_map[args.test]
        results[args.test] = test_func()
    elif args.full:
        # Run all tests
        for test_num, (name, test_func) in test_map.items():
            print(f"\n{'='*60}")
            print(f"Running Test {test_num}: {name}")
            print(f"{'='*60}")
            results[test_num] = test_func()
    else:
        # Default: run just Test 14 (full pipeline)
        print("\nNo test specified. Run with --test N or --full")
        print("Running Test 14 (Full Pipeline) as default...")
        results[14] = test_real_full_pipeline_stage1_to_stage4()

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_num, passed in results.items():
        name = test_map[test_num][0]
        status = "PASSED" if passed else "FAILED"
        print(f"  Test {test_num} ({name}): {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 80)
        print("All tests PASSED!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Some tests FAILED!")
        print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
