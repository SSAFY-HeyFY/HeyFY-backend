package com.ssafy.ssashinsa.heyfy.exchange.dto.reservation;


import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "환전 예약 요청 dto")
public class ExchangeReservationRequestDto {
    @Schema(description = "환전량(환전통화기준)", example = "100")
    @NotNull
    private Double transactionBalance;
    @Schema(description = "환전통화", example = "USD")
    @NotNull
    private String currency;
    @Schema(description = "pin", example = "123456")
    @NotNull
    private String pinNumber;
    @Schema(description = "기준 환전", example = "1330")
    @NotNull
    private Double baseExchangeRate;
}
