"""Module to parse a serialized trajectory and export it as an object fitted for learning."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional, Dict

from pddl_plus_parser.lisp_parsers.pddl_tokenizer import PDDLTokenizer
from pddl_plus_parser.models import (
    Domain,
    Observation,
    State,
    ActionCall,
    PDDLFunction,
    Predicate,
    GroundedPredicate,
    Problem,
    MultiAgentObservation,
    NOP_ACTION,
    PDDLObject,
    SignatureType,
)


class TrajectoryParser:
    """Class that is able to parse a trajectory file and extract its content.

    Note: This class assumes that the trajectory size is tractable and thus, reads the entire file at once.
    For lay loading of a trajectory future development is needed.

    Warning: if the problem object is none, all objects should be present in the initial state of the trajectory.
    Furthermore, in this case the script will not support type hierarchy or type checking for the observation objects.
    """

    partial_domain: Domain  # domain containing only the publicly known information.
    problem: Problem
    logger: logging.Logger

    def __init__(self, partial_domain: Domain, problem: Optional[Problem] = None):
        self.partial_domain = partial_domain
        self.problem = problem
        self.logger = logging.getLogger(__name__)
        self._possible_objects = None
        # When interning is enabled, maps a grounded predicate's untyped representation to a single
        # shared GroundedPredicate instance that is reused across every state it appears in.
        self._predicate_cache: Optional[Dict[str, GroundedPredicate]] = None
        # When interning is enabled, maps a grounded fluent's untyped representation to its resolved
        # signature. The signature is identical across every state a fluent appears in (only its
        # value changes), so it is resolved once and shared, skipping repeated type resolution.
        self._fluent_signature_cache: Optional[Dict[str, SignatureType]] = None

    @property
    def possible_objects(self) -> Dict[str, PDDLObject]:
        """The combined mapping of problem objects and domain constants.

        This mapping is constant for the lifetime of the parser, so it is computed once and cached
        to avoid rebuilding it for every grounded predicate / fluent that is parsed.
        """
        if self._possible_objects is None:
            self._possible_objects = {**self.problem.objects, **self.partial_domain.constants}

        return self._possible_objects

    def _read_trajectory_file(self, trajectory_file_path: Path) -> PDDLTokenizer:
        """Reads the trajectory file and exports the lines containing the data about the trajectory.

        :param trajectory_file_path: the path to the trajectory file.
        :return: the tokenizer that is able to parse the trajectory strings to expressions.
        """
        self.logger.debug(f"Reading the file - {trajectory_file_path}")
        return PDDLTokenizer(file_path=trajectory_file_path)

    def parse_state(self, state_data: List[List[Union[str, List[str]]]]) -> State:
        """Parse the trajectory's state data and extracts the state.

        :param state_data: the AST representing the state.
        :return: the state object.
        """
        self.logger.info("Parsing the observed state.")
        state_predicates = defaultdict(set)
        state_fluents = {}
        for expression in state_data:
            if expression[0] == "=":  # This is an assignment of a grounded numeric fluent.
                if len(expression) != 3:  # ['=', <function items as a list>, '<value>']
                    raise SyntaxError(
                        "A numeric fluent should be of length 3. Fluent scheme: "
                        "(= (<fluent_name> <argument>) <value>)"
                        f"Received - {expression}"
                    )

                self.logger.debug("Component found is a numeric fluent, starting to process the fluent.")
                function_data = expression[1]
                assigned_value = float(expression[2])
                numeric_fluent = self.parse_grounded_numeric_fluent(function_data)
                self.logger.debug("Setting the fluent's value to - %s", assigned_value)
                numeric_fluent.set_value(assigned_value)
                state_fluents[numeric_fluent.untyped_representation] = numeric_fluent
                continue

            if expression[0] in self.partial_domain.predicates:
                self.logger.debug("Component found is a predicate, starting to process it.")
                lifted_predicate = self.partial_domain.predicates[expression[0]]
                if self._predicate_cache is not None:
                    # Reuse a single shared instance per unique grounded fact across all states.
                    predicate_key = f"({expression[0]} {' '.join(expression[1:])})"
                    grounded_predicate = self._predicate_cache.get(predicate_key)
                    if grounded_predicate is None:
                        grounded_predicate = self.parse_grounded_predicate(expression, lifted_predicate)
                        self._predicate_cache[predicate_key] = grounded_predicate
                else:
                    grounded_predicate = self.parse_grounded_predicate(expression, lifted_predicate)

                state_predicates[lifted_predicate.untyped_representation].add(grounded_predicate)
                continue

            raise ValueError(f"Received illegal state component - {expression}")

        return State(predicates=state_predicates, fluents=state_fluents)

    def parse_grounded_numeric_fluent(self, grounded_numeric_fluent: List[str]) -> PDDLFunction:
        """Parse a single grounded numeric fluent in the problem.

        :param grounded_numeric_fluent: the grounded fluent that is present in the problem.
        :return: the function object representing the grounded fluent.
        """
        function_name = grounded_numeric_fluent[0]
        self.logger.info("Starting to parse the grounded numeric fluent - %s", function_name)
        assert function_name in self.partial_domain.functions
        lifted_function = self.partial_domain.functions[function_name]
        # For now, assuming that fluents have valid parameters.
        fluent_signature_items = grounded_numeric_fluent[1:]
        if len(fluent_signature_items) != len(lifted_function.signature):
            raise ValueError(
                f"Received fluent - {function_name} with wrong number of parameters! "
                f"Expected - {len(lifted_function.signature)} and received - {len(fluent_signature_items)}"
            )

        # A fluent's signature is identical across every state it appears in (only its value varies),
        # so when interning is enabled it is resolved once and shared, skipping the type resolution.
        if self._fluent_signature_cache is not None:
            fluent_key = f"({function_name} {' '.join(fluent_signature_items)})"
            cached_signature = self._fluent_signature_cache.get(fluent_key)
            if cached_signature is not None:
                return PDDLFunction(name=function_name, signature=cached_signature)

        if self.problem is None:
            self.logger.debug("Since we don't know the objects in the problem, we don't need to validate their types.")
            fluent_signature = {
                object_name: list(lifted_function.signature.values())[index]
                for index, object_name in enumerate(fluent_signature_items)
            }
        else:
            possible_objects = self.possible_objects
            fluent_signature = {
                object_name: possible_objects[object_name].type for object_name in fluent_signature_items
            }
            for grounded_param_type, lifted_param_type in zip(
                fluent_signature.values(), lifted_function.signature.values()
            ):
                assert grounded_param_type.is_sub_type(lifted_param_type)

        if self._fluent_signature_cache is not None:
            self._fluent_signature_cache[fluent_key] = fluent_signature

        return PDDLFunction(name=function_name, signature=fluent_signature)

    def parse_grounded_predicate(
        self, grounded_predicate_ast: List[str], lifted_predicate: Predicate
    ) -> GroundedPredicate:
        """Parse the grounded predicate that appears in the problem definition.

        :param grounded_predicate_ast: the AST that represents the grounded predicate.
        :param lifted_predicate: the lifted predicate that the domain defines.
        :return: the grounded predicate represented by the lifted predicate.
        """
        predicate_name = lifted_predicate.name
        predicate_signature_items = grounded_predicate_ast[1:]
        self.logger.info(
            "Starting the parse the grounded predicate - %s with the signature - %s.",
            predicate_name,
            predicate_signature_items,
        )
        if len(predicate_signature_items) != len(lifted_predicate.signature):
            raise ValueError(
                f"Received illegal grounded predicate with mismatching signature - {grounded_predicate_ast}"
            )

        object_mapping = {
            parameter_name: object_name
            for object_name, parameter_name in zip(predicate_signature_items, lifted_predicate.signature)
        }
        if self.problem is not None:
            object_and_consts = self.possible_objects
            grounded_signature = {
                param_name: object_and_consts[object_name].type for param_name, object_name in object_mapping.items()
            }
        else:
            grounded_signature = lifted_predicate.signature

        return GroundedPredicate(
            name=predicate_name,
            signature=grounded_signature,
            object_mapping=object_mapping,
            is_positive=True,
        )

    def parse_action_call(self, action_call_ast: List[List[str]]) -> ActionCall:
        """Parse the grounded action call in the trajectory.

        :param action_call_ast: a grounded action call in the form: [(<name> <p1> <p2> ... <pn>)]
        :return: the action call object.
        """
        self.logger.debug("Parsing the grounded action call - %s", action_call_ast)
        action_call_data = action_call_ast[0]
        return ActionCall(name=action_call_data[0], grounded_parameters=action_call_data[1:])

    def parse_joint_action(
        self, joint_action_call_ast: List[List[str]], executing_agents: List[str]
    ) -> List[ActionCall]:
        """Parse the joint action call AST in the trajectory.

        :param joint_action_call_ast: the joint action call AST in the form.
        :return: the joint action call object.
        """
        self.logger.debug(f"Parsing the joint action call - {joint_action_call_ast}")
        actions = []
        for agent_name, action_call_ast in zip(executing_agents, joint_action_call_ast):
            if action_call_ast[0] == NOP_ACTION:
                self.logger.debug(f"{agent_name} is executing a NOP action.")
                actions.append(ActionCall(name=NOP_ACTION, grounded_parameters=[]))
                continue

            actions.append(self.parse_action_call([action_call_ast]))

        return actions

    def deduce_problem_objects(
        self, initial_state_expression: List[List[Union[str, List[str]]]]
    ) -> Dict[str, PDDLObject]:
        """Deduce the problem objects from the state.

        :param initial_state_expression: the initial state expression that should contain predicates and functions that
            can be used to deduce the problem objects from.
        :return: the problem objects.
        """
        self.logger.info("Extracting the observation objects.")
        observation_objects = {}
        for expression in initial_state_expression:
            if expression[0] == "=":  # This is an assignment of a grounded numeric fluent.
                function_data = expression[1]
                function_objects = function_data[1:]
                lifted_function_signature = self.partial_domain.functions[function_data[0]].signature
                for object_name, object_type in zip(function_objects, lifted_function_signature.values()):
                    observation_objects[object_name] = PDDLObject(name=object_name, type=object_type)

                continue

            if expression[0] in self.partial_domain.predicates:
                lifted_predicate_name = expression[0]
                grounded_predicate_signature = expression[1:]
                lifted_predicate_signature = self.partial_domain.predicates[lifted_predicate_name].signature
                for object_name, object_type in zip(grounded_predicate_signature, lifted_predicate_signature.values()):
                    observation_objects[object_name] = PDDLObject(name=object_name, type=object_type)

                continue

        return observation_objects

    def parse_transition_status(self, transition_status: List[str]) -> bool:
        """Parse the transition status of the trajectory.

        :param transition_status: the transition status in the form of a list of lists.
        :return: whether the transition was successful.
        """
        self.logger.debug("Parsing the transition status.")
        success_status = transition_status[0]
        if success_status not in ["success", "failure"]:
            raise ValueError(
                f"Received illegal transition status - {success_status}. " "Expected either 'success' or 'failure'."
            )

        self.logger.debug(f"Transition status is - {success_status}.")
        return success_status == "success"

    def parse_trajectory(
        self,
        trajectory_file_path: Optional[Path] = None,
        trajectory_string: Optional[str] = None,
        executing_agents: List[str] = None,
        strict_trajectory_validation: bool = False,
        contain_transitions_status: bool = False,
        intern_predicates: bool = False,
    ) -> Union[Observation, MultiAgentObservation]:
        """Parse a trajectory and extracts the observed data into objects.

        :param trajectory_file_path: the path to the trajectory file.
        :param trajectory_string: the string representation of the trajectory.
        :param executing_agents: the list of agents that partake in the observation.
        :param strict_trajectory_validation: whether to validate the trajectory's  syntax strictly
                (mainly verify that the initial state.is labeled accordingly).
        :param contain_transitions_status: whether the trajectory contains transition status labels.
        :param intern_predicates: memory optimization for large trajectories. When enabled, identical
                grounded predicates are represented by a single shared instance across every state,
                and each next state is reused as the following component's previous state instead of
                being deep-copied. This greatly reduces memory usage, but yields a read-only
                observation: the returned states share objects, so they must not be mutated in place.
        :return: the observation extracted from the serialized trajectory.
        """
        self._predicate_cache = {} if intern_predicates else None
        self._fluent_signature_cache = {} if intern_predicates else None
        if trajectory_file_path is None and trajectory_string is None:
            raise ValueError("Either trajectory_file_path or trajectory_string should be provided.")

        if trajectory_file_path is not None and trajectory_string is not None:
            raise ValueError("Only one of trajectory_file_path or trajectory_string should be provided.")

        if trajectory_file_path is not None:
            self.logger.info("Starting to read the trajectory file!")
            tokenizer = self._read_trajectory_file(trajectory_file_path)
            observation_expression = tokenizer.parse()

        else:
            self.logger.info("Starting to parse the trajectory string!")
            tokenizer = PDDLTokenizer(pddl_str=trajectory_string)
            observation_expression = tokenizer.parse()

        observation = (
            MultiAgentObservation(executing_agents=executing_agents) if executing_agents is not None else Observation()
        )

        self.logger.debug("Parsing the initial state.")
        init_state_expression = observation_expression[0]
        if init_state_expression[0] != ":init" and strict_trajectory_validation:
            raise SyntaxError("Encountered a trajectory without an initial state!")

        if self.problem is not None:
            observation.add_problem_objects(self.problem.objects)

        else:
            self.logger.debug("Parsing the initial state and extracting the objects from the state's data.")
            observation.add_problem_objects(self.deduce_problem_objects(init_state_expression[1:]))

        previous_state = self.parse_state(init_state_expression[1:])

        self.logger.debug("Starting to generate the observation from the input trajectory.")
        state_skip_len = 2 if not contain_transitions_status else 3
        is_transition_successful = True
        for index in range(1, len(observation_expression), state_skip_len):
            macro_expression = observation_expression[index]
            if macro_expression[0] == "operator:":
                action_call = self.parse_action_call(macro_expression[1:])

            elif macro_expression[0] == "operators:":
                action_call = self.parse_joint_action(macro_expression[1:], executing_agents)

            else:
                raise SyntaxError("Encountered a trajectory without an action call!")

            if contain_transitions_status:
                macro_expression = observation_expression[index + 1]
                if macro_expression[0] != ":transition_status":
                    raise SyntaxError("Encountered a labelled trajectory without a transition status!")

                is_transition_successful = self.parse_transition_status(macro_expression[1])

            next_state_index = index + 1 if not contain_transitions_status else index + 2
            macro_expression = observation_expression[next_state_index]
            if macro_expression[0] != ":state":
                raise SyntaxError("Encountered a trajectory without a next state!")

            next_state = self.parse_state(macro_expression[1:])
            self.logger.debug("Finished parsing a trajectory component.")
            observation.add_component(
                previous_state, action_call, next_state, is_successful_transition=is_transition_successful
            )

            # When interning, share the next state directly (read-only observation) to avoid
            # duplicating every state that is both a next state and the following previous state.
            previous_state = next_state if intern_predicates else next_state.copy()

        return observation
