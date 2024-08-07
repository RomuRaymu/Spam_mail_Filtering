package net.datasa.test3.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import net.datasa.test3.domain.entity.BookEntity;

@Repository
public interface BookRepository extends JpaRepository<BookEntity, String> {

}
