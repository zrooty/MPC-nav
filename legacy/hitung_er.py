import math

def hitung_er(n, e, C_N, C_E, R):
    """
    Menghitung nilai e_r berdasarkan rumus:
    e_r = sqrt((n - C_N)^2 + (e - C_E)^2) - R

    Parameter:
    n (float): Nilai koordinat 'n'.
    e (float): Nilai koordinat 'e'.
    C_N (float): Nilai koordinat pusat 'C_N'.
    C_E (float): Nilai koordinat pusat 'C_E'.
    R (float): Nilai jari-jari atau konstanta 'R'.

    Mengembalikan:
    float: Nilai e_r.
    """

    # Hitung bagian (n - C_N)^2
    bagian_n = (n - C_N)**2

    # Hitung bagian (e - C_E)^2
    bagian_e = (e - C_E)**2

    # Hitung akar kuadrat dari jumlah kedua bagian
    jarak_pusat = math.sqrt(bagian_n + bagian_e)

    # Hitung e_r
    e_r = jarak_pusat - R

    return e_r

# --- Contoh Penggunaan ---
# Asumsi nilai-nilai sebagai contoh:
# n = 10.0
# e = 20.0
# C_N = 5.0
# C_E = 15.0
# R = 2.0

n_val = 180.0
e_val = 180.0
Cn_val = 0.0
Ce_val = 0.0
R_val = 100.0

hasil_er = hitung_er(n_val, e_val, Cn_val, Ce_val, R_val)

print(f"Nilai n: {n_val}")
print(f"Nilai e: {e_val}")
print(f"Nilai C_N: {Cn_val}")
print(f"Nilai C_E: {Ce_val}")
print(f"Nilai R: {R_val}")
print("-" * 30)
print(f"Hasil e_r adalah: {hasil_er}")