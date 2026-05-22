# DDiF Language Adaptation

DDiF (Distilling Dataset into Neural Field) uses coordinate sets and synthetic neural fields to compress dataset information.

For images, the canonical mapping is `(x, y) -> RGB`.

For language, Virgo-2 uses an experimental coordinate-field mapping:
- simplest milestone: `normalized_position -> character id distribution`
- next milestone: `[document_id, normalized_position, context_hash, scale] -> next-token distribution or latent text vector`

Language is not naturally grid-based like images, so this approach is exploratory.

Milestones:
1. Overfit tiny sequences (e.g., "hello")
2. Small corpus reconstruction
3. Next-character generation
4. Field distillation/compression with explicit budget tracking
