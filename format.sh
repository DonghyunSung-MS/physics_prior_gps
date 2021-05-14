LINT_PATHS="gps_physics/ setup.py test/"
echo ${LINT_PATHS}
isort ${LINT_PATHS}
black -l 127 ${LINT_PATHS}