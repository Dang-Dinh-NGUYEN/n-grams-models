#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

"""
Created on Thu Jul  6 19:57:28 2023

@author: hugo
"""


# on supprime manuellement l'intro, la fin (source...), la table des matieres des tomes 1 et 2, le chapitre 1 des 2tomes car il n'est pas précédé des * * * permettant de
# de distinguer le nom des chapitres du texte (on supprime pour la même raison) , on utilise ensuite une fonction pour supprimer les chapitre en utilisant ***

# enfin pour simplifier les choses  les lignes ne se terminant pas par un signe de ponctuation sont égalemnt supprimées (pour cela on a utilisé un algo pour relever l'indice de ces lignes)
# on fait cette suppression manuellement car il n'y aqu'une  vingtaine de lignes concernées

# les lignes ne peuvent se finir qu'avec liste_fin = [".","!","?",":","…","»"]


def token_list(li,
               nbr_bali):  # on applique la fonction list à notre ligne ce qui nous donne une liste de char (un seul caractere)
    """
    à appliquer à list(str) (list('bonjour à vous') produit par exemple: ['b', 'o', 'n', 'j', 'o', 'u', 'r', ' ', 'à', ' ', 'v', 'o', 'u', 's'])
    
    """
    lg = len(li)
    mettre_balise_prochain_guillemet = False
    liste_token = []
    occ_tok = ''
    for i in range(lg):

        if i == 0:  # on met la balise de début de ligne
            liste_token.extend(['<s>'] * nbr_bali)
        if li[i] == ' ':
            if occ_tok != '':  # pour éviter d'ajouter des '' dans la liste lorsqu'on rencontre  ' ? ' par exemple
                liste_token.append(occ_tok.lower())
                occ_tok = ''

        elif li[i] == '’' or li[i] == "'":
            occ_tok += li[i]
            liste_token.append(occ_tok.lower())
            occ_tok = ''

        elif li[i] == '(':  # occ_tok == '' car ( est forcément précedee d'une espace ' '
            liste_token.append('(')

        elif li[i] == ')':
            liste_token.append(occ_tok.lower())
            liste_token.append(')')
            occ_tok = ''

        elif li[i] == '-':
            liste_token.append(occ_tok.lower())
            liste_token.append('-')
            occ_tok = ''

        elif li[i] == ',':
            liste_token.append(occ_tok.lower())
            liste_token.append(',')
            occ_tok = ''

        elif li[i] == '.':
            if li[i - 1] != 'M':  # pas de pb d'indice car le pt ne se retrouvera jamais en position 0

                liste_token.append(occ_tok.lower())
                liste_token.append('.')
                occ_tok = ''
            else:  # M.
                occ_tok += '.'

            if i != lg - 1 and i != lg - 2 and li[i + 2] == '»':  # on regarde si il y a pas des guillemets aprés
                mettre_balise_prochain_guillemet = True
            elif i == lg - 1:  # si c'est le dernier caractere de la ligne on met balise de fin
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
            else:  # si pas dernier caracatere et pas de guillemets il y a une phrase qui suit donc on rajoute une balise de fin et de début
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
                liste_token.extend(['<s>'] * nbr_bali)

        elif li[i] == '!':
            liste_token.append('!')
            if i != lg - 1 and i != lg - 2 and li[i + 2] == '»':
                mettre_balise_prochain_guillemet = True
            elif i != lg - 1 and li[
                i + 1] == "…":  # il arrive que !… soit rencontré et dans ce cas , seulement le … doit mettre la balise
                pass
            elif i == lg - 1:
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
            else:
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
                liste_token.extend(['<s>'] * nbr_bali)

            # pas besoin de réinitialiser occ_tok car le ! est forcément entre deux espaces donc occ_tok == ''

        elif li[i] == '?':
            liste_token.append('?')
            if i != lg - 1 and i != lg - 2 and li[i + 2] == '»':
                mettre_balise_prochain_guillemet = True
            elif i != lg - 1 and li[
                i + 1] == "…":  # il arrive que ?… soit rencontré et dans ce cas , seulement le … doit mettre la balise
                pass
            elif i == lg - 1:
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
            else:
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
                liste_token.extend(['<s>'] * nbr_bali)

        elif li[i] == '…':
            if occ_tok != '':  # occ_tok=='' seulement quand … est précédé d'autres types de point !… ou ?… par ex
                liste_token.append(occ_tok.lower())
            liste_token.append('…')
            occ_tok = ''
            if i != lg - 1 and i != lg - 2 and li[i + 2] == '»':
                mettre_balise_prochain_guillemet = True
            elif i == lg - 1:
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
            else:
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
                liste_token.extend(['<s>'] * nbr_bali)


        elif li[
            i] == ':' and i == lg - 1:  # on traite ':' differemment seulement si ils sont en fin de ligne dans les autres cas ils sont traités comme n'importe quelle occurrence
            liste_token.append(':')  # ils ne sont jamais suivis de » fermants donc pas besoin
            liste_token.extend(['</s>'] * nbr_bali + ['\n'])

        elif li[i] == "»":
            liste_token.append("»")
            if mettre_balise_prochain_guillemet:  # si vrai cela veut dire que le guillemet est précédé d'un pt quelconque donc balise de fin de phrase
                liste_token.extend(['</s>'] * nbr_bali + ['\n'])
                mettre_balise_prochain_guillemet = False
                if i != lg - 1:  # si le guillemet n'est pas en derniere position alors il est forcément suivi d'autres occurences donc on met une balise de début
                    liste_token.extend(['<s>'] * nbr_bali)


        else:
            occ_tok += li[
                i]  # tant qu'on ne recontre pas un séparateur mis dans les conditions ci dessus on réagrége les occurrences caractere par caractere
    return liste_token


def token_ligne(ligne, nbr_bali):
    """
    prend en parametre la ligne directement obtenue avec le readline et le nbr de balise à mettre et return la ligne tokenisée sous forme d'une chaine de caractere splittable

    """
    token_str = ''
    ligne = ligne.strip()
    splitted_ligne = ligne.split()
    lg_li = len(splitted_ligne)
    guill = False
    for i in range(lg_li):
        occ = splitted_ligne[i].lower()
        lg_occ = len(occ)
        if i == 0:
            token_str += (nbr_bali - 1) * '<s> ' + '<s>'

        token_str += ' '
        for j in range(lg_occ):

            if occ[j] == "'" or occ[j] == "’" or occ[j] == "(":
                token_str += occ[j] + ' '

            elif occ[j] == "-":
                token_str += ' ' + occ[j] + ' '

            elif occ[j] == ")" or occ[j] == ",":
                token_str += ' ' + occ[j]

            elif occ[j] == ".":
                if occ[j - 1] == "M":
                    token_str += occ[j]
                else:

                    try:
                        splitted_ligne[i + 1]
                    except IndexError:

                        token_str += ' . ' + nbr_bali * '</s> ' + "\n"
                    else:
                        if splitted_ligne[i + 1] == "\"" or splitted_ligne[i + 1] == "»":
                            token_str += ' . '
                            guill = True
                        else:
                            token_str += ' . ' + nbr_bali * '</s> ' + "\n" + (nbr_bali - 1) * '<s> ' + '<s>'

            elif occ[j] == ':':
                if i == lg_li - 1:
                    token_str += ': ' + nbr_bali * '</s> ' + "\n"
                else:
                    token_str += ':'

            elif occ[j] == '!' or occ[j] == '?':
                try:
                    occ[j + 1]
                except IndexError:
                    if i == lg_li - 1:
                        token_str += occ[j] + ' ' + nbr_bali * '</s> ' + "\n"
                    else:
                        token_str += occ[j] + ' ' + nbr_bali * '</s> ' + "\n" + (nbr_bali - 1) * '<s> ' + '<s>'
                else:
                    token_str += occ[j] + ' '

            elif occ[j] == "…":
                try:
                    splitted_ligne[i + 1]
                except IndexError:
                    token_str += ' … ' + nbr_bali * '</s> ' + "\n"
                else:
                    if splitted_ligne[i + 1] == "\"" or splitted_ligne[i + 1] == "»":
                        guill = True
                        token_str += ' … '
                    else:
                        token_str += ' … ' + nbr_bali * '</s> ' + "\n" + (nbr_bali - 1) * '<s> ' + '<s>'

            elif occ[j] == "\"" or occ[j] == "»":
                token_str += occ[j]
                if guill:
                    if i == lg_li - 1:
                        token_str += ' ' + nbr_bali * '</s> ' + "\n"
                        guill = False
                    else:
                        token_str += ' ' + nbr_bali * '</s> ' + "\n" + (nbr_bali - 1) * '<s> ' + '<s>'
            else:
                token_str += occ[j]
    return token_str


def tokenizer(fichier, fichier_décriture, nbr_bali):
    compte = 0
    nbr_bali = int(nbr_bali)
    with open(fichier_décriture, 'w') as fi:
        pass  # on écrase le fichier d'arrivée au début
    with open(fichier, 'r') as fi:
        ligne = fi.readline()
        while ligne != "":  # rajouter à la fin ligne = fi.readline()
            compte += 1
            if ligne == "\n":  # si c'est un caractere saut de ligne : on ne fait rien
                pass
            else:
                ligne = ligne[:-1]  # on supprime le '\n'
                liste_char = list(ligne)
                liste_tokenisée = token_list(liste_char,
                                             nbr_bali)  # on a désormais une liste compltement tokenisée on peut donc écrire dans le fichier
                with open(fichier_décriture, 'a') as fich_ecrit:
                    for occ in liste_tokenisée:
                        if occ != '\n':
                            fich_ecrit.write(occ + ' ')
                        else:
                            fich_ecrit.write(occ)
                    # fich_ecrit.write("\n")

            ligne = fi.readline()


def tokenizer2(fichier_texte_brut, fichier_ecriture, nbr_bali):
    nbr_bali = int(nbr_bali)
    with open(fichier_ecriture, 'w', encoding="utf-8-sig", errors='ignore') as fi_ecr:
        pass
    with open(fichier_texte_brut, 'r', encoding="utf-8-sig", errors='ignore') as fi:
        ligne = fi.readline()
        while ligne != '':
            if ligne == '\n':
                pass
            else:
                ligne_tok = token_ligne(ligne, nbr_bali)
                with open(fichier_ecriture, 'a') as fi_ecr:
                    fi_ecr.write(ligne_tok)
            ligne = fi.readline()

"""
# %%

if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "fichier_texte_brut", " fichier_ecriture", "nbr_de_balise_debut_fin(int)")
    exit(0)
tokenizer2(sys.argv[1], sys.argv[2], sys.argv[3])

# %%
"""