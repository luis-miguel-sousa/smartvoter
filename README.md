# The Smart Voter

IMO, hat should be served to the knowledge base is similarly-shaped JSONs. This is an example (CHEGA just for fun!):

json```
{
  "party_name": "CHEGA",
  "key_issues": [
    {
      "issue_name": "Family and Society",
      "description": "CHEGA emphasizes strengthening family as a core institution and proposes creating a Ministry of Family to reinforce moral, civic, and economic aspects of family life.",
      "proposals": [
        {
          "title": "Creation of the Ministry of Family",
          "details": "Establish a dedicated ministry to oversee the reconstruction of family values across various areas of governance.",
          "ideology_tags": ["conservative", "family_values"]
        }
      ]
    },
    ...
  ]
```

## FAQ

### Who's running, according to the CNE:

- (AD) CDS - Partido Popular
- (AD) Partido Social Democrata
- (AD) Partido Popular Monárquico
- Partido Socialista
- (CDU) Partido Comunista Português
- (CDU) Partido Ecologista Os Verdes
- Partido Comunista dos Trabalhadores Portugueses
- Ergue-te
- Partido da Terra
- Bloco de Esquerda
- Partido Trabalhista Português
- PESSOAS – ANIMAIS – NATUREZA
- Movimento Alternativa Socialista
- LIVRE
- Juntos pelo Povo
- ALTERNATIVA DEMOCRÁTICA NACIONAL
- Nós, Cidadãos!
- Partido Unido dos Reformados e Pensionistas
- Iniciativa Liberal
- Aliança
- CHEGA
- Reagir Incluir Reciclar
- Volt Portugal

### Who has 2024 sorted out already?

Livre, Volt... none of the big ones :)

### What's done?

- IL 2022 (BS4 scraper in a notebook) -> has a JSON, not in the structure above.
- Livre 2024 (BS4 scraper in a notebook) -> has a JSON, not in the structure above.
- Chega 2021 (just served their very elementary document as plaintext to ChatGPT) -> has a JSON, in the structure above

### Any other findings?

Generating the common structure document all at once won't work with ChatGPT. Claude 2.1 supports 200K tokens, but it's a little bit more suspicious (who's gonna validate the JSON?)
