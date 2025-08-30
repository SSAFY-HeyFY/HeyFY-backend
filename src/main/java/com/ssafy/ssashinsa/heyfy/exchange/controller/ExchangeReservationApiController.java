package com.ssafy.ssashinsa.heyfy.exchange.controller;

import com.ssafy.ssashinsa.heyfy.authentication.annotation.AuthUser;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.exchange.docs.ExchangeReservationDocs;
import com.ssafy.ssashinsa.heyfy.exchange.domain.ExchangeReservation;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationRequestDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.service.ExchangeReservationService;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@Tag(name = "환전 예약 api")
@RestController
@RequiredArgsConstructor
@RequestMapping("/exchange/reservation")
public class ExchangeReservationApiController {
    private final ExchangeReservationService exchangeReservationService;

    @PostMapping
    @ExchangeReservationDocs
    public ExchangeReservationResponseDto makeReservation(@AuthUser UserDetails userDetails, @RequestBody @Valid ExchangeReservationRequestDto requestDto){
        log.info("환전 예약 요청 들어옴");
        try{
            ExchangeReservation exchangeReservation = exchangeReservationService.createExchangeReservation(userDetails.getUsername(),requestDto);
        } catch (CustomException e){
            if(e.getErrorCode().equals(AuthErrorCode.INVALID_PIN_NUMBER)){
                ExchangeReservationResponseDto response = ExchangeReservationResponseDto.builder()
                        .success(false)
                        .build();
                return response;
            }
            throw e;
        }


        return ExchangeReservationResponseDto.builder()
                .success(true)
                .build();
    }
}
