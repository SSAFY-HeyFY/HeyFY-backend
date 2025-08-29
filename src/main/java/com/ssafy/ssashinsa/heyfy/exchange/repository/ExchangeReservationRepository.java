package com.ssafy.ssashinsa.heyfy.exchange.repository;

import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeReservation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface ExchangeReservationRepository extends JpaRepository<ExchangeReservation, Long> {
    @Query("SELECT e FROM ExchangeReservation e WHERE e.exchangeCompleted = false")
    List<ExchangeReservation> findAllNotCompleted();
    @Query("SELECT e FROM ExchangeReservation e JOIN FETCH e.user WHERE e.exchangeCompleted = false")
    List<ExchangeReservation> findAllNotCompletedWithUser();
    @Query("SELECT DISTINCT e FROM ExchangeReservation e " +
           "JOIN FETCH e.user u " +
           "LEFT JOIN FETCH u.fcmTokens " +
           "WHERE e.exchangeCompleted = false")
    List<ExchangeReservation> findAllNotCompletedWithUserAndFcmTokens();

    @Modifying
    @Query("UPDATE ExchangeReservation e SET e.exchangeCompleted = true WHERE e.id IN :ids")
    int bulkComplete(@Param("ids") List<Long> ids);
}
