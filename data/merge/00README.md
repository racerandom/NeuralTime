## 変換プログラム
### timex2offset.rb

1文字単位に TIMEX3 と EVENT タグの情報の情報を割り当てる

CREATION_TIME のみ例外として付与しない

$ for i in BCCWJ-TIMEX/*.xml; do ruby timex2offset.rb $i > tmp/`basename $i| cut -c1-5`-timex.txt ; done

### ud2offset.rb

1文字単位に 行番号を割り当てる

$ for i in UD_Japanese/*.conll; do ruby ud2offset.rb $i > tmp/`basename $i| cut -c1-5`-ud.txt ; done

$ for i in UD_Japanese/*.conll; do cp $i tmp/`basename $i| cut -c1-5`-ud.conll ; done

### merge.rb

$ for i in tmp/*.conll; do ruby merge.rb tmp/`basename $i -ud.conll`-timex.txt tmp/`basename $i .conll`.txt $i > merged/`basename $i`; done