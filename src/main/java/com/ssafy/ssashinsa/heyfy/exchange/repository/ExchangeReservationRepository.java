package com.ssafy.ssashinsa.heyfy.exchange.repository;

import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeReservation;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ExchangeReservationRepository extends JpaRepository<ExchangeReservation, Long> {
}
