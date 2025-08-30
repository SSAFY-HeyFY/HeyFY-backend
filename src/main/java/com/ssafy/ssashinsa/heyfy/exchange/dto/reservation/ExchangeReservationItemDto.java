package com.ssafy.ssashinsa.heyfy.exchange.dto.reservation;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Schema(description = "환전 예약 내역 아이템 dto")
public class ExchangeReservationItemDto {
    @Schema(description = "환전 예약 id", example = "1")
    private Long reservationId;
    @Schema(description = "환전 통화", example = "USD")
    private String currency;
    @Schema(description = "환전 금액(환전통화기준)", example = "100.0")
    private Double amount;
    @Schema(description = "예약 기준환율", example = "1300.5")
    private Double baseExchangeRate;
    @Schema(description = "예약 생성 시간", example = "2023-10-01T12:34:56")
    private LocalDateTime createdAt;
    @Schema(description = "환전 완료 여부", example = "false")
    private boolean exchangeCompleted;
    @Schema(description = "환전 취소 여부", example = "false")
    private boolean canceled;
}
