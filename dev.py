# %%

from pii_redaction.detection import detect_pii_with_gemma

# %%

text = "Contact Jane Doe (born 1/5/1985) at 123 Oak St, Springfield, IL. Her email is j.doe@email.net and SSN is 987-65-4321."

pii_list = detect_pii_with_gemma(text)

# %%

pii_list
