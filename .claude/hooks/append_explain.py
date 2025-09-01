#!/usr/bin/env python3

import json
import sys

def main() -> None:
    try:
        data = json.loads(sys.stdin.read())
        prompt = data.get("prompt", "")
        if not prompt:
            raise ValueError("No prompt found")
        if prompt.rstrip().endswith("-e"):
            print(
                "\n above is the relevent logs, your jobs is to: \n"
                "think harder about these logs to say \n"
                "and give me a simpler or short explanation \n"
                "DO NOT JUMP INTO CONCLUSION! DO NOT MAKE ASSUMPTIONS! QUITE YOUR EGO\n"
                "AND ASSUME YOU KNOW NOTHING\n"
                "now you explain the logs to me, and make suggestions for next steps and why\n"
                "answer in short"
            )
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()