# reset_models.sh - Reset model files to force redownload

echo "WARNING: This will remove all downloaded models and cached files."
read -p "Do you want to continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing model files..."
    rm -rf models/seamless_m4t/*
    rm -rf models/indictrans2/*
    rm -rf models/xtts_v2/*
    
    # Keep the directory structure
    mkdir -p models/seamless_m4t
    mkdir -p models/indictrans2
    mkdir -p models/xtts_v2/speakers
    
    echo "Model files removed. They will be redownloaded on next application start."
else
    echo "Operation cancelled."
fi