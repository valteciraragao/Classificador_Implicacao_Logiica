# gera_dataset.py
rules = [
    # Adição
    ("(p)->(p|q)", "Tautologia"),
    ("(p)->(p|r)", "Tautologia"),
    # Simplificação
    ("(p&q)->p", "Tautologia"),
    ("(p&r)->r", "Tautologia"),
    # Silogismo Disjuntivo
    ("((p|q)&(~p))->q", "Tautologia"),
    ("((r|p)&(~r))->p", "Tautologia"),
    # Modus Ponens
    ("((p)&(p->q))->q", "Tautologia"),
    ("((r)&(r->s))->s", "Tautologia"),
    # Modus Tollens
    ("((~q)&(p->q))->(~p)", "Tautologia"),
    ("((~s)&(r->s))->(~r)", "Tautologia"),
    # Contingência genérica
    ("(p)->q", "Contingência"),
    ("(q)->r", "Contingência"),
    # Contradição pura
    ("(p)&(~p)", "Contradição"),
    ("(q)&(~q)", "Contradição"),
]

# Gere variações com permutações de variáveis
import itertools
vars_ = ['p','q','r']
for combo in itertools.permutations(vars_, 3):
    p,q,r = combo
    # adição
    rules.append((f"({p})->({p}|{q})", "Tautologia"))
    # simplificação
    rules.append((f"({p}&{q})->{p}", "Tautologia"))
    # contingência
    rules.append((f"({p})->{q}", "Contingência"))

# Salvar X e y
X = [expr for expr, _ in rules]
y = [label for _, label in rules]

# Escreva em disco para inspeção
with open("dataset_expressões.txt","w") as f:
    for expr, label in rules:
        f.write(f"{expr}\t{label}\n")
print(f"Dataset gerado: {len(X)} exemplos")
