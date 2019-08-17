for file in jsons/*; do
    echo python main.py "$(basename "$file")"
    python main.py "$(basename "$file")"
    python ckpt_to_graph.py "$(basename "$file")"
done