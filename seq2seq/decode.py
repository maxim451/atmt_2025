import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
from seq2seq.beam import BeamSearch, BeamSearchNode

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens

def decode_beam_search(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor,
                       max_out_len: int, tgt_tokenizer: spm.SentencePieceProcessor,
                       args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    batch_size = src_tokens.size(0)
    assert batch_size == 1, "Beam search decoding currently supports batch_size=1 only"

    beam_search = BeamSearch(beam_size=beam_size, max_len=max_out_len, pad=PAD)

    initial_input = torch.tensor([[BOS]], dtype=torch.long, device=device)
    trg_pad_mask = (initial_input == PAD).unsqueeze(1).unsqueeze(2)

    output = model(src_tokens, src_pad_mask, initial_input, trg_pad_mask)
    log_probs = torch.nn.functional.log_softmax(output[:, -1, :], dim=-1)

    top_log_probs, top_indices = log_probs[0].topk(beam_size)
    for i in range(beam_size):
        token = top_indices[i].unsqueeze(0).unsqueeze(0)
        logp = top_log_probs[i].item()
        sequence = torch.cat([initial_input, token], dim=1)
        node = BeamSearchNode(search=beam_search, emb=None, lstm_out=None,
                              final_hidden=None, final_cell=None, mask=None,
                              sequence=sequence, logProb=logp, length=1)
        beam_search.add(-node.eval(alpha), node)

    for _ in range(max_out_len - 1):
        current_beams = beam_search.get_current_beams()
        if not current_beams:
            break

        for score, node in current_beams:
            if node.sequence[0, -1].item() == EOS:
                beam_search.add_final(score, node)
                continue

            trg_pad_mask = (node.sequence == PAD).unsqueeze(1).unsqueeze(2)
            output = model(src_tokens, src_pad_mask, node.sequence, trg_pad_mask)
            log_probs = torch.nn.functional.log_softmax(output[:, -1, :], dim=-1)

            top_log_probs, top_indices = log_probs[0].topk(beam_size)
            for i in range(beam_size):
                token = top_indices[i].unsqueeze(0).unsqueeze(0)
                logp = top_log_probs[i].item()
                new_seq = torch.cat([node.sequence, token], dim=1)
                new_node = BeamSearchNode(search=beam_search, emb=None, lstm_out=None,
                                          final_hidden=None, final_cell=None, mask=None,
                                          sequence=new_seq, logProb=node.logp + logp,
                                          length=node.length + 1)
                beam_search.add(-new_node.eval(alpha), new_node)

        beam_search.prune()

    best_score, best_node = beam_search.get_best()
    output_seq = best_node.sequence.squeeze().tolist()

    if BOS in output_seq:
        output_seq = output_seq[1:]
    if EOS in output_seq:
        output_seq = output_seq[:output_seq.index(EOS)+1]

    return [output_seq]

