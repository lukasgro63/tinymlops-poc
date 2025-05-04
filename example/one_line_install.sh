#!/bin/bash
# Ein-Zeilen-Installationssskript für TinyMLOps

# Zeige Banner
echo -e "\033[0;34m===================================================================\033[0m"
echo -e "\033[0;34m       TinyMLOps Einfache Ein-Zeilen-Installation                 \033[0m"
echo -e "\033[0;34m===================================================================\033[0m"

# Führe das Setup durch
curl -sSL https://raw.githubusercontent.com/lukasgro63/tinymlops-poc/main/example/setup_simple.sh | bash

# Ende
echo "Installation abgeschlossen!"