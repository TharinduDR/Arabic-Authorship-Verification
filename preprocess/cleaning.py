def remove_tashkil(word):
    damma = "ُ"
    sukun = "ْ"
    fatha = "َ"
    kasra = "ِ"
    shadda = "ّ"
    tanweendam = "ٌ"
    tanweenfath = "ً"
    tanweenkasr = "ٍ"
    tatweel = "ـ"

    tashkil = (damma, sukun, fatha, kasra, shadda, tanweendam, tanweenfath, tanweenkasr, tatweel)

    w = [letter for letter in word if letter not in tashkil]
    return "".join(w)


def clean_arabic(x):
    return remove_tashkil(x)


def normalize(some_string):
    normdict = {
        'ة': 'ه',
        'أ': 'ا',
        'إ': 'ا',
        'ي': 'ى',

    }
    out = [normdict.get(x, x) for x in some_string]
    return ''.join(out)
