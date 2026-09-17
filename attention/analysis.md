# Analysis of BERT Attention Heads

## Head 1: Layer 1, Head 11 (Determiner to Associated Noun)
- **Description**: This attention head appears to learn the relationship between determiners (such as "the", "a") and the specific nouns they refer to. The determiner assigns strong attention weight directly to the corresponding noun that follows it in the noun phrase.
- **Example 1**: "We visited the old [MASK] yesterday." (The token "the" directs high attention toward the noun position).
- **Example 2**: "She found a silver [MASK] in the garden." (The token "a" directs attention toward the subsequent noun).

## Head 2: Layer 6, Head 4 (Prepositions to Noun Phrase Objects)
- **Description**: This attention head captures the structural connection between prepositions and the core noun of the prepositional complement phrase. Prepositions attend strongly to the noun governed by them, helping determine spatial or situational relations.
- **Example 1**: "He placed the glass on the wooden [MASK]." (The preposition "on" focuses on the head of the object noun phrase).
- **Example 2**: "They walked through a dark [MASK] at night." (The preposition "through" focuses attention onto the completing noun token).
