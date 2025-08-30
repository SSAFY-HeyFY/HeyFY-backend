package com.ssafy.ssashinsa.heyfy.exchange.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationCancelResponseDto;
import com.ssafy.ssashinsa.heyfy.exchange.dto.reservation.ExchangeReservationResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "환전 예약 취소")
@Documented
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "취소 성공",
                content = @Content(schema = @Schema(implementation = ExchangeReservationCancelResponseDto.class))),
})
public @interface ExchangeReservationCancelDocs {
}
