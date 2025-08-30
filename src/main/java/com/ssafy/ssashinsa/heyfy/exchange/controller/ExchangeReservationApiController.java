package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.authentication.annotation.AuthUser;
import com.ssafy.ssashinsa.heyfy.exchange.docs.ExchangeReservationDocs;
import com.ssafy.ssashinsa.heyfy.exchange.docs.ExchangeReservationListDocs;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeReservation;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationItemDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationListDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeReservationService;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@Tag(name = "환전 예약 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange/reservation")
public class ExchangeReservationApiController {
    private final ExchangeReservationService exchangeReservationService;

    @PostMapping
    @ExchangeReservationDocs
    public ResponseEntity<ExchangeReservationResponseDto> makeReservation(@AuthUser UserDetails userDetails, @RequestBody @Valid ExchangeReservationRequestDto requestDto){
        log.info("환전 예약 요청 들어옴");
        ExchangeReservation exchangeReservation = exchangeReservationService.createExchangeReservation(userDetails.getUsername(),requestDto);
        return ResponseEntity.ok(ExchangeReservationResponseDto.builder()
                .success(true)
                .build());
    }

    @GetMapping
    @ExchangeReservationListDocs
    public ResponseEntity<ExchangeReservationListDto> getReservations(@AuthUser UserDetails userDetails){
        List<ExchangeReservation> result =  exchangeReservationService.getExchangeReservations(userDetails.getUsername());
        List<ExchangeReservationItemDto> data = result.stream().map(reservation -> {
            return ExchangeReservationItemDto.builder()
                    .reservationId(reservation.getId())
                    .currency(reservation.getDepositAccountCurrency().toString())
                    .amount(reservation.getTransactionBalance())
                    .createdAt(reservation.getCreatedAt())
                    .baseExchangeRate(reservation.getBaseExchangeRate())
                    .exchangeCompleted(reservation.isExchangeCompleted())
                    .canceled(reservation.isCanceled())
                    .build();
        }).toList();

        return ResponseEntity.ok(ExchangeReservationListDto.builder()
                .data(data)
                .build());
    }
}
