def flatten_list(nested_list: list):
    total_list = []
    for maybe_list in nested_list:
        if isinstance(maybe_list, list):
            total_list.extend(flatten_list(maybe_list))
        else:
            total_list.append(maybe_list)
    return total_list


def char_count(s: str):
    dict = {}
    for char in s:
        if char in dict:
            dict[char] += 1
        else:
            dict[char] = 1
    return dict