#!/usr/bin/env bash
# $ alias envy="source ~/Workspace/repos/scratchpad/envy/envy.sh"

main() {
    if ! [ -z "${VIRTUAL_ENV}" ]; then
        echo "Deactivating venv (${VIRTUAL_ENV})"
        deactivate
        return 0
    fi

    local venv_name
    venv_name=${1:-.venv}

    local selected_dir
    local venv_dir

    selected_dir=$(pwd)
    for i in $(seq $(dir_depth ${selected_dir})); do
        # Don't activate venvs in /home/user, or any dirs that that are less than this depth
        if [ $(dir_depth "${selected_dir}") -le 2 ]; then break; fi

        venv_dir="${selected_dir}/${venv_name}"
        if [ -d "${venv_dir}" ]; then
            echo "Found ${venv_dir}"
            source "${venv_dir}/bin/activate"
            return 0
        fi
        selected_dir=$(dirname ${selected_dir})
    done

    echo "Couldn't find a venv named ${venv_name}, creating one in current directory"
    python -m venv ${venv_name}
    venv_dir="$(pwd)/${venv_name}"
    source "${venv_dir}/bin/activate"
    return 0
}

dir_depth() {
    echo "${1}" | grep -o "/" | wc -l
}

main "$@" || exit 1
