set -euo pipefail

func() {
    return 0
}

foo=$(func 2>/dev/null || echo $?)
if [ -z $foo ]; then foo=0; fi

echo ${foo}
