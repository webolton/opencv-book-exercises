from sklearn.feature_extraction.text import CountVectorizer

data = [
    'I am Mohammed Abacha, the son of the late Nigerian Head of '
    'State who died on the 8th of June 1998. Since i have been '
    'unsuccessful in locating the relatives for over 2 years now '
    'I seek your consent to present you as the next of kin so '
    'that the proceeds of this account valued at US$15.5 Million '
    'Dollars can be paid to you. If you are capable and willing '
    'to assist, contact me at once via email with following '
    'details: 1. Your full name, address, and telephone number. '
    '2. Your Bank Name, Address. 3.Your Bank Account Number and '
    'Beneficiary Name - You must be the signatory.'
]

vec = CountVectorizer()
x = vec.fit_transform(data)

vec.get_feature_names()[:5]
x.to_array()[0, :5]
