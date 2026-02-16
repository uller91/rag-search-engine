def check_tokens(tkns1, tkns2) -> bool:
    for tk1 in tkns1:
        for tkn2 in tkns2:
            if tk1 in tkn2:
                return True
    return False