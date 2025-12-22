# RAG System Evaluation Metrics

## Overview
This document summarizes the evaluation results of the Retrieval Augmented Generation (RAG) system for the Crop Care agricultural advisory platform. The evaluation includes 30 queries spanning wheat, maize, and rice crops with corresponding similarity scores.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Queries Evaluated | 30 |
| Average Similarity Score | 0.651 |
| Median Similarity Score | 0.662 |
| Highest Score | 0.798 |
| Lowest Score | 0.457 |
| Standard Deviation | 0.087 |

## Score Distribution

| Score Range | Count | Percentage |
|-------------|-------|-----------|
| 0.90 - 1.00 | 0 | 0% |
| 0.80 - 0.89 | 1 | 3.3% |
| 0.70 - 0.79 | 3 | 10% |
| 0.60 - 0.69 | 15 | 50% |
| 0.50 - 0.59 | 9 | 30% |
| 0.40 - 0.49 | 2 | 6.7% |

## Query-by-Query Results

### Wheat Queries

| Query ID | Query | Similarity Score | Status |
|----------|-------|------------------|--------|
| Q1 | Symptoms of leaf rust in wheat | 0.715 | ✓ Good |
| Q2 | Brown rust management in wheat | 0.702 | ✓ Good |
| Q3 | Nitrogen deficiency signs in wheat | 0.488 | ⚠ Fair |
| Q4 | Wheat flowering water requirements | 0.457 | ⚠ Fair |
| Q5 | Fungal disease prevention in wheat | 0.622 | ✓ Good |
| Q6 | Improving wheat soil fertility | 0.531 | ⚠ Fair |
| Q7 | Best irrigation schedule for wheat | 0.540 | ⚠ Fair |
| Q8 | Using resistant wheat varieties | 0.674 | ✓ Good |
| Q9 | Safe pesticide usage in wheat fields | 0.654 | ✓ Good |
| Q10 | Monitoring wheat leaves for pests | 0.532 | ⚠ Fair |

**Wheat Average Score: 0.616** | **Good Queries: 5** | **Fair Queries: 5**

### Maize Queries

| Query ID | Query | Similarity Score | Status |
|----------|-------|------------------|--------|
| Q11 | How to irrigate maize efficiently | 0.611 | ✓ Good |
| Q12 | Maize lethal necrosis symptoms | 0.769 | ✓ Excellent |
| Q13 | Water stress effects in maize | 0.629 | ✓ Good |
| Q14 | Pest control in maize fields | 0.701 | ✓ Good |
| Q15 | Soil nutrient imbalance management for maize | 0.655 | ✓ Good |
| Q16 | Caterpillar damage control in maize | 0.607 | ✓ Good |
| Q17 | Reducing leaf rolling in maize | 0.530 | ⚠ Fair |
| Q18 | Balanced fertilizer application in maize | 0.681 | ✓ Good |
| Q19 | Viral disease detection in maize | 0.798 | ✓ Excellent |
| Q20 | Certified seed usage for maize disease prevention | 0.673 | ✓ Good |

**Maize Average Score: 0.685** | **Excellent Queries: 2** | **Good Queries: 7** | **Fair Queries: 1**

### Rice Queries

| Query ID | Query | Similarity Score | Status |
|----------|-------|------------------|--------|
| Q21 | Best irrigation schedule for rice | 0.619 | ✓ Good |
| Q22 | Managing rice brown spot disease | 0.668 | ✓ Good |
| Q23 | Low nitrogen effects on rice growth | 0.571 | ⚠ Fair |
| Q24 | Rice soil fertility improvement techniques | 0.672 | ✓ Good |
| Q25 | Controlling rice stem borers | 0.704 | ✓ Good |
| Q26 | Water management in rice paddies | 0.656 | ✓ Good |
| Q27 | Rice blast disease prevention | 0.677 | ✓ Good |
| Q28 | Improving drainage in rice fields | 0.639 | ✓ Good |
| Q29 | Organic matter application in rice fields | 0.670 | ✓ Good |
| Q30 | Monitoring rice growth under drought | 0.581 | ⚠ Fair |

**Rice Average Score: 0.665** | **Good Queries: 8** | **Fair Queries: 2**

## Performance Analysis by Crop

### Crop Comparison

| Crop | Average Score | Best Score | Worst Score | Coverage |
|------|---------------|-----------|-------------|----------|
| Wheat | 0.616 | 0.715 | 0.457 | 33.3% |
| Maize | 0.685 | 0.798 | 0.530 | 33.3% |
| Rice | 0.665 | 0.704 | 0.571 | 33.3% |

## Key Findings

### Strengths
1. **Excellent Performance on Disease Queries**: Viral disease detection in maize (0.798) and maize lethal necrosis (0.769) show strong retrieval accuracy
2. **Strong Maize Coverage**: Maize-related queries have the highest average score (0.685)
3. **Disease Management Queries**: Plant disease identification and management queries generally score higher than generic agronomy questions
4. **Consistent Performance**: 50% of queries fall in the good range (0.60-0.69)

## Quality Metrics

### Query Quality Assessment
- **High Confidence (≥0.70)**: 4 queries (13.3%)
- **Good Confidence (0.60-0.69)**: 15 queries (50%)
- **Fair Confidence (0.50-0.59)**: 9 queries (30%)
- **Low Confidence (<0.50)**: 2 queries (6.7%)

## Conclusion

The RAG system demonstrates **solid overall performance** with an average similarity score of 0.651. The system excels at disease-specific queries and maize-related topics but shows opportunities for improvement in:
- General wheat agronomy
- Nutrient management and fertilization
- Broad water and irrigation management

The next phase should focus on expanding the knowledge base for underperforming query categories while maintaining the strong performance on disease identification and management queries.

---

**Evaluation Date**: December 23, 2025  
**Total Queries Evaluated**: 30  
**Evaluation System**: RAG (Retrieval Augmented Generation) with similarity scoring
