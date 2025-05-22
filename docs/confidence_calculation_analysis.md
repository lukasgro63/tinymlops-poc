# LightweightKNN Confidence Calculation - Code Analysis

This document analyzes the actual confidence calculation implementation in TinyLCM's LightweightKNN classifier, showing how it goes beyond simple vote counting to provide enhanced confidence measures for drift detection.

## Actual Code Implementation

### Location
**File**: `/tinylcm/core/classifiers/knn.py`  
**Method**: `predict_proba()` (lines 300-469)

### Key Implementation Details

#### 1. Confidence Scaling Factor
```python
# Line 372 in knn.py
confidence_scaling = 100.0  # Dramatically increased to make confidence more sensitive to distance
```

**Where to Configure**: This is currently **hardcoded** in the `predict_proba()` method. To make it configurable, you would need to:

1. **Add it as a constructor parameter**:
```python
def __init__(
    self,
    k: int = 5,
    distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
    max_samples: int = 100,
    use_numpy: bool = True,
    weight_by_distance: bool = False,
    tie_break_by_time: bool = True,
    confidence_scaling: float = 100.0  # Add this parameter
):
    # ... existing code ...
    self.confidence_scaling = confidence_scaling
```

2. **Use it in predict_proba**:
```python
# Replace line 372
confidence_scaling = self.confidence_scaling
```

#### 2. Complete Confidence Calculation Algorithm

```python
# Lines 401-441 in knn.py
def calculate_enhanced_confidence(neighbors, k):
    # Count votes and distances by class
    class_stats = {}
    for neighbor_idx, distance in neighbors:
        label = self.y_train[neighbor_idx]
        if label not in class_stats:
            class_stats[label] = {"count": 0, "total_distance": 0.0}
        class_stats[label]["count"] += 1
        class_stats[label]["total_distance"] += distance

    # Calculate adjusted probabilities
    total_adjusted_vote = 0.0
    adjusted_votes = {}

    for label, vote_info in class_stats.items():
        # Base vote probability (traditional KNN)
        vote_prob = vote_info["count"] / k
        
        # Average distance for this class
        class_avg_distance = vote_info["total_distance"] / vote_info["count"]
        
        # Distance factor: higher distances = lower confidence
        # Uses smooth decay function (sigmoid-like)
        distance_factor = 1.0 / (1.0 + confidence_scaling * class_avg_distance)
        
        # Final adjusted vote combines vote count and distance
        adjusted_vote = vote_prob * distance_factor
        adjusted_votes[label] = adjusted_vote
        total_adjusted_vote += adjusted_vote

    # Normalize to get final probabilities
    if total_adjusted_vote > 0:
        probabilities = {
            label: vote / total_adjusted_vote 
            for label, vote in adjusted_votes.items()
        }
    else:
        # Fallback to simple vote counting if adjustment fails
        votes = Counter([self.y_train[neighbor[0]] for neighbor in neighbors])
        probabilities = {label: count / k for label, count in votes.items()}
    
    return probabilities
```

### 3. Mathematical Breakdown

#### Traditional KNN Confidence:
```
P(class) = vote_count / k
```

#### Enhanced TinyLCM Confidence:
```
# Step 1: Calculate base vote probability
vote_prob = vote_count / k

# Step 2: Calculate average distance for class
class_avg_distance = total_distance / vote_count

# Step 3: Apply distance decay function
distance_factor = 1.0 / (1.0 + confidence_scaling * class_avg_distance)

# Step 4: Combine vote probability with distance factor
adjusted_vote = vote_prob * distance_factor

# Step 5: Normalize across all classes
P(class) = adjusted_vote / sum(all_adjusted_votes)
```

### 4. Configuration Options

#### Current Implementation:
- **Hardcoded**: `confidence_scaling = 100.0`
- **Location**: Line 372 in `predict_proba()` method

#### To Make Configurable:

**Option 1: Constructor Parameter (Recommended)**
```python
class LightweightKNN(BaseAdaptiveClassifier):
    def __init__(self, ..., confidence_scaling: float = 100.0):
        self.confidence_scaling = confidence_scaling
```

**Option 2: Configuration File Integration**
```json
{
  "adaptive_classifier": {
    "type": "LightweightKNN",
    "k": 5,
    "distance_metric": "euclidean",
    "confidence_scaling": 100.0,
    "max_samples": 200
  }
}
```

**Option 3: Runtime Setter Method**
```python
def set_confidence_scaling(self, scaling_factor: float):
    """Update confidence scaling factor at runtime."""
    self.confidence_scaling = scaling_factor
    logger.info(f"Confidence scaling updated to {scaling_factor}")
```

### 5. Impact of Scaling Factor Values

#### Low Values (1.0 - 10.0):
- **Effect**: Confidence less sensitive to distance changes
- **Use Case**: When distances naturally vary widely
- **Behavior**: More traditional KNN-like confidence

#### Medium Values (10.0 - 100.0):
- **Effect**: Balanced sensitivity to both votes and distances
- **Use Case**: Standard operation (current default: 100.0)
- **Behavior**: Good compromise between vote counting and distance weighting

#### High Values (100.0+):
- **Effect**: Confidence drops rapidly with increased distance
- **Use Case**: When drift detection needs high sensitivity
- **Behavior**: Strong penalization of distant neighbors

### 6. Actual Code Comments

From the source code comments:
```python
# Line 309-314 comments:
"""
The confidence calculation is enhanced to consider distance information:
- Traditional KNN just counts class votes among neighbors
- This implementation scales votes by distance, so samples that are further away
  result in lower confidence scores, even if the predicted class is the same
- This is important for drift detection, as it allows confidence to drop
  when new samples drift away from the training distribution
"""

# Line 369-372 comments:
"""
Distance-based confidence scaling factor
- Higher values make confidence drop more quickly with distance
- Lower values make confidence more resilient to distance changes
"""
```

### 7. Debugging and Monitoring

The code includes comprehensive logging:
```python
# Line 461-463: Confidence statistics logging
logger.info(f"KNN ENHANCED CONFIDENCE STATS (v2) - Avg distance: {avg_distance:.6f}, Avg max probability: {avg_max_proba:.6f}")
if n_samples == 1:  # Show detailed probabilities for single predictions
    logger.info(f"DETAILED PROBAS: {probas[0]}")
```

### 8. Integration with Drift Detection

The enhanced confidence feeds into drift detection systems:
```python
# Confidence monitors can use these enhanced probabilities
class ConfidenceMonitor(AutonomousDriftDetector):
    def update(self, record):
        confidence = record.get('confidence')  # This comes from enhanced predict_proba
        if confidence < self.confidence_threshold:
            # Trigger drift detection based on low confidence
            return True, {'reason': 'low_confidence', 'value': confidence}
```

## Summary

The confidence calculation in TinyLCM's LightweightKNN is **currently hardcoded** at line 372 with `confidence_scaling = 100.0`. To make it configurable, you would need to add it as a constructor parameter and update the configuration system to pass this value during classifier initialization.

The enhanced confidence calculation provides superior drift detection capabilities by considering both vote counts and distances, making it more sensitive to samples that drift away from the training distribution.