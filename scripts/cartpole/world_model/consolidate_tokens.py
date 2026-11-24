"""Copy the patch tokens of an embeddings directory into one flat memmap.

``embed.py --store-tokens`` writes tokens per episode, which cannot be memory
mapped; patch-level world models sample random windows across all episodes from
the flat cache this writes (see ``wm.token_cache``).
"""

import argparse
import time
from pathlib import Path

from world_model_playground.wm.token_cache import consolidate_tokens, open_token_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        nargs="+",
        required=True,
        help="directories written by embed.py --store-tokens",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing flat cache"
    )
    args = parser.parse_args()
    for embeddings_dir in args.embeddings_dir:
        start = time.perf_counter()
        output = consolidate_tokens(embeddings_dir, overwrite=args.overwrite)
        store = open_token_store(embeddings_dir)
        print(
            f"{output}: {store.tokens.shape} {store.tokens.dtype} "
            f"({store.tokens.nbytes / 1e9:.1f} GB, {time.perf_counter() - start:.0f}s)"
        )


if __name__ == "__main__":
    main()
