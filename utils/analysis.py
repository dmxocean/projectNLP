from collections import defaultdict
import random  # Added for random selection

def find_examples_and_patterns(true_labels, pred_labels, observations, is_bio=False):
    """Find success/error examples and common error patterns at sentence level"""
    success_examples = []
    error_examples = []
    error_patterns = defaultdict(int)
    
    for i, (true_seq, pred_seq, obs_seq) in enumerate(zip(true_labels, pred_labels, observations)):
        # Split into sentence chunks
        sentence_chunks = []
        current_chunk = []
        current_chunk_len = 0
        
        for j, (true, pred, obs) in enumerate(zip(true_seq, pred_seq, obs_seq)):
            current_chunk.append({
                "token": obs[0] if isinstance(obs, tuple) else obs,
                "pos": obs[1] if isinstance(obs, tuple) else None,
                "true": true,
                "pred": pred,
                "is_correct": true == pred
            })
            current_chunk_len += 1
            
            # Record error patterns
            if true != pred and j > 0:
                # Store components separately for better formatting later
                pattern_key = (true_seq[j-1], true, pred_seq[j-1], pred)
                error_patterns[pattern_key] += 1
            
            # Create sentence chunks (approximate sentences)
            if current_chunk_len >= 10 or j == len(true_seq) - 1:
                # Check if this chunk has errors or successes
                has_errors = any(not token["is_correct"] for token in current_chunk)
                has_successes = any(token["is_correct"] and token["true"] != "O" for token in current_chunk)
                
                if has_errors or has_successes:
                    sentence_chunks.append({
                        "id": f"{i}-{len(sentence_chunks)}",
                        "tokens": current_chunk,
                        "has_errors": has_errors,
                        "has_successes": has_successes,
                        "errors": sum(1 for t in current_chunk if not t["is_correct"]),
                        "successes": sum(1 for t in current_chunk if t["is_correct"] and t["true"] != "O")
                    })
                
                current_chunk = []
                current_chunk_len = 0
        
        # Add sentence chunks to appropriate lists
        for chunk in sentence_chunks:
            if chunk["has_errors"]:
                error_examples.append(chunk)
            if chunk["has_successes"]:
                success_examples.append(chunk)
    
    # Select random examples instead of sorting
    random.shuffle(error_examples)
    random.shuffle(success_examples)
    
    # Format error patterns for display with proper alignment
    formatted_patterns = []
    for (prev_true, curr_true, prev_pred, curr_pred), count in sorted(
        error_patterns.items(), key=lambda x: x[1], reverse=True
    )[:8]:
        true_transition = f"{prev_true}->{curr_true}"
        pred_transition = f"{prev_pred}->{curr_pred}"
        formatted_patterns.append((true_transition, pred_transition, count))
    
    return {
        "success": success_examples[:5],  # Show more examples since they're shorter sentences
        "errors": error_examples[:5],
        "patterns": formatted_patterns
    }

def format_vertical_example(example, include_pos=False, show_comparison=False):
    """Format example for vertical display with one token per line"""
    rows = []
    
    if show_comparison:
        rows.append(f"Example ID: {example['id']}")
        rows.append("-" * 70)
        rows.append(f"{'TOKEN':<20} {'TRUE':<12} {'FIRST-ORDER':<12} {'SECOND-ORDER':<12} {'STATUS':<10}")
        rows.append("-" * 70)
        
        for token in example["tokens"]:
            word = token["token"]
            pos = token["pos"] if include_pos and token["pos"] else ""
            token_text = f"{word}/{pos}" if pos else word
            token_text = token_text[:18] + '..' if len(token_text) > 20 else token_text
            
            status = ""
            if token["improved"]:
                status = "IMPROVED"
            elif token["regressed"]:
                status = "REGRESSED"
                
            rows.append(f"{token_text:<20} {token['true']:<12} {token['first']:<12} {token['second']:<12} {status:<10}")
    else:
        rows.append(f"Example ID: {example['id']}")
        rows.append("-" * 70)
        rows.append(f"{'TOKEN':<20} {'TRUE':<12} {'PREDICTED':<12} {'CORRECT':<8}")
        rows.append("-" * 70)
        
        for token in example["tokens"]:
            word = token["token"]
            pos = token["pos"] if include_pos and token["pos"] else ""
            token_text = f"{word}/{pos}" if pos else word
            token_text = token_text[:18] + '..' if len(token_text) > 20 else token_text
            
            rows.append(f"{token_text:<20} {token['true']:<12} {token['pred']:<12} {'T' if token['is_correct'] else 'F':<8}")
    
    return "\n".join(rows)

def analyze_transitions(true_labels, pred_labels):
    """Analyze transition accuracies"""
    transitions = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for j in range(1, len(true_seq)):
            transition = f"{true_seq[j-1]}->{true_seq[j]}"
            transitions[transition]["total"] += 1
            if true_seq[j] == pred_seq[j]:
                transitions[transition]["correct"] += 1
    
    # Calculate accuracy and sort
    results = {}
    for trans, counts in transitions.items():
        if counts["total"] >= 5:  # Only include common transitions
            results[trans] = {
                "accuracy": counts["correct"] / counts["total"],
                "correct": counts["correct"],
                "total": counts["total"]
            }
    
    return sorted(results.items(), key=lambda x: x[1]["total"], reverse=True)[:8]

def analyze_bio_transitions(true_labels, pred_labels):
    """Analyze BIO-specific transitions"""
    transitions = {
        "B->I correct": 0,
        "B->I incorrect": 0,
        "I->I correct": 0,
        "I->I incorrect": 0,
        "O->B correct": 0,
        "O->B incorrect": 0,
        "B->O incorrect": 0,  # Should continue as I instead of O
        "I->O incorrect": 0,  # Should continue as I instead of O
    }
    
    for i, (true_seq, pred_seq) in enumerate(zip(true_labels, pred_labels)):
        for j in range(1, len(true_seq)):
            # B->I transitions (start of entity continuing)
            if true_seq[j-1].startswith('B-') and true_seq[j].startswith('I-'):
                same_entity = true_seq[j-1][2:] == true_seq[j][2:]
                if same_entity and pred_seq[j].startswith('I-') and pred_seq[j][2:] == true_seq[j][2:]:
                    transitions["B->I correct"] += 1
                else:
                    transitions["B->I incorrect"] += 1
                    if pred_seq[j] == 'O':
                        transitions["B->O incorrect"] += 1
            
            # I->I transitions (middle of entity continuing)
            if true_seq[j-1].startswith('I-') and true_seq[j].startswith('I-'):
                same_entity = true_seq[j-1][2:] == true_seq[j][2:]
                if same_entity and pred_seq[j].startswith('I-') and pred_seq[j][2:] == true_seq[j][2:]:
                    transitions["I->I correct"] += 1
                else:
                    transitions["I->I incorrect"] += 1
                    if pred_seq[j] == 'O':
                        transitions["I->O incorrect"] += 1
            
            # O->B transitions (start of new entity)
            if true_seq[j-1] == 'O' and true_seq[j].startswith('B-'):
                if pred_seq[j].startswith('B-') and pred_seq[j][2:] == true_seq[j][2:]:
                    transitions["O->B correct"] += 1
                else:
                    transitions["O->B incorrect"] += 1
    
    return transitions

def compare_with_first_order(true_labels, second_order_preds, first_order_preds, observations):
    """Compare second-order model with first-order model at sentence level"""
    improvements = []
    regressions = []
    
    improve_count = 0
    regress_count = 0
    
    for i, (true_seq, first_pred, second_pred, obs_seq) in enumerate(zip(
            true_labels, first_order_preds, second_order_preds, observations)):
        
        # Split into sentence chunks (roughly 8-15 tokens per sentence)
        sentence_chunks = []
        current_chunk = []
        current_chunk_len = 0
        chunk_improvements = 0
        chunk_regressions = 0
        
        for j, (true, first, second, obs) in enumerate(zip(true_seq, first_pred, second_pred, obs_seq)):
            # Track where second-order improved or regressed
            improved = False
            regressed = False
            
            if first != second:
                if second == true and first != true:
                    improve_count += 1
                    chunk_improvements += 1
                    improved = True
                elif second != true and first == true:
                    regress_count += 1
                    chunk_regressions += 1
                    regressed = True
            
            current_chunk.append({
                "token": obs[0] if isinstance(obs, tuple) else obs,
                "pos": obs[1] if isinstance(obs, tuple) else None,
                "true": true, 
                "first": first,
                "second": second,
                "improved": improved,
                "regressed": regressed
            })
            current_chunk_len += 1
            
            # Create sentence chunks
            if current_chunk_len >= 10 or j == len(true_seq) - 1:
                if chunk_improvements > 0 or chunk_regressions > 0:
                    sentence_chunks.append({
                        "id": f"{i}-{len(sentence_chunks)}",
                        "tokens": current_chunk,
                        "improvements": chunk_improvements,
                        "regressions": chunk_regressions
                    })
                
                current_chunk = []
                current_chunk_len = 0
                chunk_improvements = 0
                chunk_regressions = 0
        
        # Add chunks to appropriate lists
        for chunk in sentence_chunks:
            if chunk["improvements"] > 0:
                improvements.append(chunk)
            if chunk["regressions"] > 0:
                regressions.append(chunk)
    
    # Select random examples instead of sorting
    random.shuffle(improvements)
    random.shuffle(regressions)
    
    return {
        "improvements": improvements[:5],
        "regressions": regressions[:5],
        "improve_count": improve_count,
        "regress_count": regress_count
    }