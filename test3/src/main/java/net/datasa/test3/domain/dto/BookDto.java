package net.datasa.test3.domain.dto;

import java.time.LocalDate;

import lombok.Data;
import lombok.RequiredArgsConstructor;

@Data
@RequiredArgsConstructor
public class BookDto {
	String ISBN;
	String title;
	String author;
	String publisher;
	LocalDate publishDate;
	Integer price;
	Float discountRate;
}
