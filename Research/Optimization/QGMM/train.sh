for file in jsons/*multiple_2*; do
    echo python main.py "$(basename "$file")"
    #python main.py "$(basename "$file")"
    python main_multiple.py "$(basename "$file")"
done