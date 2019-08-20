# coding: cp932
require 'rubygems'
require 'diff/lcs'

timex_chars = Array.new
timex_tags = Array.new
File.open(ARGV[0]).each do |line|
  line.chomp!
  array = line.split("\t")
  timex_chars.push array[0]  # TIMEX 側の文字列
  timex_tags.push array[1]   # TIMEX 側のタグ情報
end

ud_chars = Array.new
ud_lines = Array.new
File.open(ARGV[1]).each do |line|
  line.chomp!
  array = line.split("\t")
  ud_chars.push array[0]     # UD 側の文字列
  ud_lines.push array[1]     # UD 側の行番号
end

timex2ud = Hash.new
sdiffs = Diff::LCS.sdiff(timex_chars,ud_chars)
sdiffs.each do |sdiff|
  sdiff_array = sdiff.to_a
#  p sdiff_array
  timex2ud[sdiff_array[1][0]] = sdiff_array[2][0] # TIMEX側の文字列番号と UD側の文字列番号の割り当て
end

ud_lines2timex_tags = Hash.new
timex_tags.each_with_index do |tag,i|
  
  # TIMEX 側のタグ情報　TIMEX側の文字列番号　UD 側の文字列番号　UD側の行番号
  unless tag.nil?
    ud_lines2timex_tags[ud_lines[timex2ud[i]].to_i] = tag
#    puts tag.to_s + "\t" + i.to_s + "\t" + timex2ud[i].to_s + "\t" + ud_lines[timex2ud[i]]
  end
end

linenum = 0
File.open(ARGV[2]).each do |line|
  line.chomp!
  linenum += 1
  if ud_lines2timex_tags.key? linenum
    puts line + "\t" + ud_lines2timex_tags[linenum]
  else
    puts line + "\t"
  end
end

