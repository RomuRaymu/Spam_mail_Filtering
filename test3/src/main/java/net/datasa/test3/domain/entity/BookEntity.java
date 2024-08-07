package net.datasa.test3.domain.entity;

import java.time.LocalDate;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;

@Data
@Entity
@Table(name="books")
public class BookEntity {
	@Id
	@Column(name = "ISBN", nullable = false, length = 20)
	private String ISBN;
	
	@Column(name = "title", nullable = false, length = 200)
	private String title;
	
	@Column(name = "author", nullable = false, length = 100)
	private String author;
	
	@Column(name = "publisher", nullable = false, length = 100)
	private String publisher;
	
	@Column(name = "publishDate")
	private LocalDate publishDate;
	
	@Column(name = "price")
	private Integer price;
	
	@Column(name = "discountRate")
	private Float discountRate;
}
