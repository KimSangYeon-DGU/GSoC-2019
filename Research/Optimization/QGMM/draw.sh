for file in models/*dist*; do
    echo python main.py "$(basename "$file")"
    python ckpt_to_graph.py "$(basename "$file")"
done