# Qwen3.5 native full-attention-8 self injection

This is a same-model functional upper-bound experiment. Qwen3.5-4B prefills
Context, retains gold-supporting-token pre-RoPE K/native V from all eight native
full-attention layers, and injects them without a Writer into the same frozen
Qwen3.5-4B Receiver.

The 24 DeltaNet layers receive no Context convolution or recurrent state. Only
q_proj/o_proj LoRA at the eight full-attention layers is trained with answer CE.
Question never enters the Sender.

Final conditions: question only, gold supporting text, self sparse KV with LoRA
off/on, shuffled self KV, and forced-zero self KV. No hard gate is enforced.
