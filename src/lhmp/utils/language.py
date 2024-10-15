from typing import Optional, Union
from collections import Counter
import re


def join_to_sentence(string_list: Union[list, Counter]) -> str:
    """
    Join a list of strings into a sentence, with commas and "and" appropriately placed
    if the input is a list, it is simply joined with commas
    if the input is an object of type Counter, the number of occurences of each element is added to the string
    """
    if isinstance(string_list, Counter):
        help_string_list = []
        for key, value in string_list.items():
            if value == 1:
                help_string_list.append(f"1 {key}")
            else:
                if key.endswith("s"):
                    help_string_list.append(f"{value} {key}")
                else:
                    help_string_list.append(f"{value} {key}s")

        string_list = help_string_list

    if len(string_list) == 1:
        return string_list[0]
    elif len(string_list) == 2:
        return " and ".join(string_list)
    else:
        return ", ".join(string_list[:-1]) + ", and " + string_list[-1]


def extract_from_within(text: str, a: str, b: str) -> str:
    match = re.findall(rf"{a}(.*?){b}", text, re.DOTALL)[0]
    if "\n" in match:
        match = match.replace("\n", "")
    return match
