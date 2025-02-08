# Multi-Example Meta Prompt Implementation Plan

## Overview
This document outlines the plan to update the MetaPromptGraph to support processing multiple examples sequentially, ensuring that a single system message produces acceptable outputs for all examples.

## Goals
- Handle multiple examples in `state['examples']` iteratively.
- Ensure the same system message produces outputs meeting acceptance criteria for all examples.
- Use the current graph as a subgraph for multi-example processing.
- Maintain the current AgentState structure without creating new state classes.
- Invalidate previous outputs and checks when the system message changes.

## Implementation Plan

### 1. Create Outer Graph for Multi-Example Processing

#### Nodes:
1. START: Initial node to begin processing
2. INITIALIZE: Set up the state for the first example
3. PROCESS_EXAMPLE: Run the existing MetaPromptGraph for a single example
4. UPDATE_SYSTEM_MESSAGE: Update the system message if necessary
5. VALIDATE_ALL_EXAMPLES: Check if the current system message works for all examples
6. CHECK_COMPLETION: Determine if processing is complete
7. END: Final node to conclude processing

#### Edges:
1. START -> INITIALIZE
2. INITIALIZE -> PROCESS_EXAMPLE
3. PROCESS_EXAMPLE -> UPDATE_SYSTEM_MESSAGE
4. UPDATE_SYSTEM_MESSAGE -> VALIDATE_ALL_EXAMPLES
5. VALIDATE_ALL_EXAMPLES -> CHECK_COMPLETION
6. CHECK_COMPLETION -> PROCESS_EXAMPLE (if not all examples pass)
7. CHECK_COMPLETION -> END (if all examples pass or recursion limit reached)

### 2. Implement Node Functions

#### INITIALIZE:
- Reset necessary state variables
- Set `current_example_index` to 0
- Initialize system message if not present

#### PROCESS_EXAMPLE:
- Run the existing MetaPromptGraph for the current example
- Return the updated state

#### UPDATE_SYSTEM_MESSAGE:
- Update the system message if the current example produced a better result
- If updated, invalidate all previous outputs and checks

#### VALIDATE_ALL_EXAMPLES:
- Use the current system message to generate outputs for all examples
- Check if all outputs meet their respective acceptance criteria
- Store results for each example

#### CHECK_COMPLETION:
- Determine if all examples have passed their acceptance criteria
- Check if the recursion limit has been reached
- If not all examples pass, select an example that did not meet the criteria and set it as the current example
- Decide whether to continue processing or end the graph

### 3. Create `run_multi_example_meta_prompt()` Method
- Implement the outer graph structure
- Define the logic for each node
- Handle the flow between nodes

### 4. Update AgentState
- Add `current_example_index` to AgentState
- Add `example_results` to store pass/fail status for each example
- Modify `AgentState.to_dict()` to use the current example based on the index

### 5. Modify `run_meta_prompt_graph()`
- Ensure compatibility with `current_example_index` in the state
- Maintain existing functionality for single-example processing

### 6. Update `__call__` Method
- Check for presence of multiple examples
- Call appropriate method based on number of examples (single or multiple)

### 7. Add Logging and Error Handling
- Implement logging for multi-example processing progress
- Handle potential errors specific to multi-example scenarios

### 8. Testing and Validation
- Create test cases with multiple examples
- Verify functionality in various scenarios (all pass, some fail, recursion limit reached)
- Ensure system message changes trigger re-validation of all examples

### 9. Optimization and Refinement
- Analyze performance of multi-example processing
- Refine implementation based on performance analysis
- Consider future enhancements (e.g., parallel processing options)

## Implementation Steps

### Stage 1: Workflow Structure

#### Stage 1.1: Initialize Multi-Example Support
1. Update AgentState:
   - Add `current_example_index` and `example_results`
   - Modify `AgentState.to_dict()` for current example selection

Estimated new code: ~20 lines
Estimated modified code: ~10 lines

Suggested test modifications:
- Update existing tests to work with new AgentState structure
- Add test for `AgentState.to_dict()` with multiple examples

Estimated test code changes: ~30 lines

#### Stage 1.2: Implement INITIALIZE Node
1. Create INITIALIZE node function:
   - Reset state variables
   - Set current_example_index to 0
   - Initialize system message if not present

Estimated new code: ~25 lines
Estimated modified code: ~5 lines

Suggested new tests:
- Test INITIALIZE node function
- Test initialization of multi-example state

Estimated new test code: ~40 lines

#### Stage 1.3: Implement UPDATE_SYSTEM_MESSAGE Node
1. Create UPDATE_SYSTEM_MESSAGE node function:
   - Update system message if current example produced better result
   - Invalidate previous outputs and checks if updated

Estimated new code: ~30 lines
Estimated modified code: ~5 lines

Suggested new tests:
- Test UPDATE_SYSTEM_MESSAGE node function
- Test system message update and invalidation of previous results

Estimated new test code: ~50 lines

#### Stage 1.4: Implement VALIDATE_ALL_EXAMPLES Node
1. Create VALIDATE_ALL_EXAMPLES node function:
   - Generate outputs for all examples using current system message
   - Check if outputs meet respective acceptance criteria
   - Store results for each example

Estimated new code: ~40 lines
Estimated modified code: ~5 lines

Suggested new tests:
- Test VALIDATE_ALL_EXAMPLES node function
- Test validation of multiple examples

Estimated new test code: ~60 lines

#### Stage 1.5: Implement CHECK_COMPLETION Node
1. Create CHECK_COMPLETION node function:
   - Determine if all examples have passed or recursion limit reached
   - Select next example to process if not all passed
   - Decide whether to continue processing or end the graph

Estimated new code: ~35 lines
Estimated modified code: ~5 lines

Suggested new tests:
- Test CHECK_COMPLETION node function
- Test different scenarios (all passed, some failed, recursion limit)

Estimated new test code: ~70 lines

#### Stage 1.6: Create Multi-Example Graph Structure
1. Implement new graph structure:
   - Add new nodes: START, INITIALIZE, PROCESS_EXAMPLE, UPDATE_SYSTEM_MESSAGE, VALIDATE_ALL_EXAMPLES, CHECK_COMPLETION, END
   - Create edges between nodes
   - Implement conditional logic for CHECK_COMPLETION node

Estimated new code: ~50 lines
Estimated modified code: ~10 lines

Suggested new tests:
- Test overall multi-example graph structure
- Test graph execution with multiple examples

Estimated new test code: ~100 lines

### Stage 2: Multi-Example Handling
2. Create `run_multi_example_meta_prompt()` method:
   - Implement the outer graph logic
   - Handle state transitions and example processing

3. Update AgentState:
   - Add `current_example_index` and `example_results`
   - Modify `AgentState.to_dict()` for current example selection

### Stage 3: Integration and Testing
4. Modify `run_meta_prompt_graph()`:
   - Adapt to handle `current_example_index`

5. Update `__call__` method:
   - Add logic to detect and handle multiple examples

### Stage 4: Logging and Error Handling
6. Enhance logging and error handling:
   - Add logging for multi-example processing progress
   - Implement error handling for new scenarios

### Stage 5: Testing and Optimization
7. Develop test cases:
   - Create diverse scenarios for testing
   - Implement tests to verify multi-example functionality and system message updates

8. Perform optimization and refinement:
   - Analyze and improve performance
   - Consider future enhancements

## Conclusion
This updated plan provides a comprehensive approach to implementing multi-example support in the MetaPromptGraph. It ensures that a single system message produces acceptable outputs for all examples before concluding the process, and invalidates previous results when the system message changes. This approach maintains the integrity of the multi-example processing while leveraging the existing MetaPromptGraph for individual example processing.