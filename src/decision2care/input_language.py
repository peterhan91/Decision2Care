"""Reject non-English scripts at model boundaries, allowing standard micrometre units."""
import unicodedata


def require_english_input(value, path='input'):
    # This catches accidental Chinese/UI text even after JSON unicode escaping.
    # It is a script guard, not a semantic English-language classifier.
    if isinstance(value, str):
        # Greek mu and the micro sign in micrometre units are valid in English
        # scientific text. Only the validation copy changes; raw input is preserved.
        check=value.replace('μm','micrometres').replace('µm','micrometres')
        if any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in check):
            raise ValueError(f'English model input required at {path}; non-Latin text found.')
    elif isinstance(value, dict):
        for key, item in value.items():
            require_english_input(key, path+'.<key>')
            require_english_input(item, path+'.'+str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            require_english_input(item, f'{path}[{i}]')
