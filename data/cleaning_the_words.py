import csv
from pathlib import Path
import argparse

TURKISH_LOWER_MAP = str.maketrans({
    "I": "ı",
    "İ": "i",
})

def turkish_lower(text: str) -> str:
    return text.translate(TURKISH_LOWER_MAP).lower()

def temizle_ve_ayristir(lines: list[str]) -> list[str]:
    seen = set()
    sonuc = []

    for line in lines:
        # Satırdaki kelimeleri Türkçe kurallarına göre küçült ve boşluklardan böl
        for parca in turkish_lower(line).split():
            if parca and parca not in seen:
                seen.add(parca)
                sonuc.append(parca)

    return sonuc

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV dosyasındaki kelime sütununu okur, küçültür, böler ve tekrarları siler."
    )
    # Varsayılan dosya ismini CSV dosyanıza göre güncelledik
    parser.add_argument("input", nargs="?", default="words.csv", help="Girdi CSV dosyası")
    parser.add_argument("output", nargs="?", default="cleaned_words.txt", help="Çıktı dosyası")
    
    # args=[] diyerek Jupyter Notebook gibi ortamlardaki gizli argüman hatalarını engelliyoruz
    args, unknown = parser.parse_known_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1. YENİ KISIM: Sadece CSV'nin "kelime" sütununu okuma
    lines = []
    with open(input_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Eğer sütun başlığı tam olarak 'kelime' ise ve içi boş değilse listeye ekle
            if row.get('kelime'):
                lines.append(row['kelime'])

    # 2. Okunan kelimeleri temizle ve tekrarlardan arındır
    sonuc = temizle_ve_ayristir(lines)
    
    # 3. Sonuçları çıktı dosyasına alt alta yazdır
    output_path.write_text("\n".join(sonuc) + "\n", encoding="utf-8")
    
    print(f"İşlem tamam! Toplam {len(sonuc)} adet benzersiz kelime '{args.output}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()