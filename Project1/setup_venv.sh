#!/bin/bash
# python -m venv venv
"C:/Users/P3856387/AppData/Local/Programs/Python/Python313/python.exe" -m venv venv
source venv/Scripts/activate
pip install --upgrade pip setuptools
pip install scikit-learn pandas numpy matplotlib seaborn onnx onnxmltools onnxruntime joblib
pip freeze > requirements.txt

which python
which pip


# with CLI
# cat << EOF > setup_venv.sh
# write the script
# type EOF in the last line to create the file 

# cat           = Command to output text (concatenate and print)
# << EOF        = here-document start - tells cat to read everything until the next EOF
# >             = Redirects the output of cat to a file instead of the terminal.
# setup_venv.sh = The filename where the input (everything until EOF) is saved